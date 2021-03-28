import argparse
import pickle
from os import path

import torch
from tqdm import tqdm

from losses.knn_loss import KNNLoss
from network.events_net import EventsNetCPC
from stream_generators.StreamGeneratorSingleEventSubset import StreamGeneratorSingleEventSubset
from stream_generators.dataset import get_custom_dataset
from utils.MatplotlibUtils import reduce_dims_and_plot
from utils.batch_iterator import Batch


def get_args():
    parser = argparse.ArgumentParser(description="""
	=============================
		Mine CEP Patterns
	=============================
	""")
    # Subset miner
    parser.add_argument('-bn_algorithm', default='exact',
                        help='algorithm to use for bayesian network learning')
    parser.add_argument('-num_events_per_samples', default=5, type=int,
                        help='Number of events to use for each bayesian network sample')
    parser.add_argument('-min_bond', default=0.15, type=float,
                        help='minimum bond value for the events CORI proposals generator')
    parser.add_argument('-min_support', default=0.05, type=float,
                        help='minimum support value for the events CORI proposals generator')
    parser.add_argument('-timeout', type=int,
                        help='Amount of time to run the anytime algorithm. If not provided, the algorithm '
                             'will run until all possible subsets are sampled, or a user entered the stop code (stop)')
    parser.add_argument('-seq_len', default=20, type=int,
                        help='Length of time window from slicing the stream')
    parser.add_argument('-conditions_seq_len', default=30, type=int,
                        help='Length of time window from slicing the stream')
    parser.add_argument('-conditions_min_bond', default=0.65, type=float,
                        help='minimum bond value for the conditions CORI proposals generator')
    parser.add_argument('-num_processes', default=8, type=int,
                        help='Maximum number of processes to run simultaneously during BN creations')
    parser.add_argument('-single_combinations', action='store_true', default=False,
                        help='use only last occurrence when analyzing a subsequence')
    parser.add_argument('-memory_efficient', action='store_true', default=False,
                        help='Whether to dump plots and csvs or not')

    # Random stream
    parser.add_argument('-stream_length', default=500000, type=int,
                        help='define the minimum length of the stream to be generated')
    parser.add_argument('-num_events', type=int, default=40,
                        help='Number of events in the randomly generated stream')
    parser.add_argument('-avg_attrs', type=int, default=5,
                        help='Average number of attributes per event in the randomly generated stream')
    # parser.add_argument('-max_attrs', type=int, default=8,
    # 					help='Maximum number of attributes per event in the randomly generated stream')
    parser.add_argument('-patterns_count', type=int,
                        help='Number of patterns to generate in the randomly generated stream')
    parser.add_argument('-conditions_count', type=int,
                        help='Number of conditions to generate in the randomly generated stream')
    parser.add_argument('-conditions_per_pattern', type=int,
                        help='Number of conditions per pattern')
    parser.add_argument('-noise_prob', type=float, default=0.3,
                        help='probability of noisy event injection')

    # Pickled stream
    parser.add_argument('-stream_file',
                        help='Path to a file containing a pickled stream')

    # Demo stream
    parser.add_argument('-dump_stream', action='store_true', default=False,
                        help='dump the current stream to a pickle')
    parser.add_argument('-path_stream_to_dump', type=str, default='stream.file',
                        help='path to dump the current stream to a pickle')
    parser.add_argument('-demo_file', default=r'not_a_path.py',
                        help='Path to a file containing conditions for stream generation')

    # IO
    parser.add_argument('-exps_dir', default=r'exps',
                        help='Path of directory to save the logs and the results')
    parser.add_argument('-log_file', type=str, default='log.log',
                        help='path to write the log')
    parser.add_argument('-plot_correlations', action='store_true', default=False,
                        help='whether to plot sasu and basu matrices')
    parser.add_argument('-plot_graphs', action='store_true', default=False,
                        help='whether to plot bn graph')

    # Conditions
    parser.add_argument('-exclude_unaries', action='store_true', default=False,
                        help='Exclude unary conditions')
    parser.add_argument('-skip_conjuction', action='store_true', default=False,
                        help='skip the last step, which is creating conjunction of simple conditions')
    parser.add_argument('-topk', type=int, default=25,
                        help='Amount of complex rules to extract')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_events = 40
    length = 300000
    stream_path = f'stream_{length}.file'
    if not path.exists(stream_path):
        stream = StreamGeneratorSingleEventSubset(
            provided_conditions=None,
            provided_events=None,
            provided_patterns=None,
            use_time_conditions=False,
            use_attr_conditions=True,
            min_length=length,
            num_events=num_events,
            min_attributes=30,
            max_attributes=150,
            start_pattern_prob=1,
            continue_pattern_prob=1,
            patterns_count=10,
            conditions_count=10,
            subset=list(range(num_events))
        )
        data = [e for e in tqdm(stream)]
        with open(stream_path, 'wb') as fp:
            pickle.dump(stream, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"Loading stream from {stream_path}")
        with open(stream_path, 'rb') as fp:
            stream = pickle.load(fp)
        print("Loading done")

    dataset = get_custom_dataset(stream, stream.data, seq_len=stream.max_window, return_base_index=False, stride=1)
    loader = Batch(dataset, batch_size=2000, shuffle=True, return_last=False)
    model = EventsNetCPC(events_attributes=[e.n_attrs for e in stream.event_types],
                         embedding_dim=10,
                         context_dim=5,
                         samples_to_predict=1,)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3,
                                 weight_decay=5e-4)
    loss_fn = KNNLoss(k=500)
    n_epochs = 100
    log_every = 10

    delta_improvement = 1
    old_loss = float("inf")

    for epoch in range(n_epochs):
        new_loss = 0
        for i_batch, batch in enumerate(loader):
            optimizer.zero_grad()
            y = model(batch)
            loss = loss_fn(y)
            new_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i_batch % log_every == 0:
                print(f"Epoch {epoch + 1}: {i_batch} / {len(loader)}  loss = {loss.item()}")

        new_loss = new_loss / len(loader)
        d = abs(new_loss - old_loss)

        print(f"New loss: {new_loss}    delta: {d}")
        print(f"Require delta of less than {delta_improvement} to stop training")

        if d < delta_improvement:
            break

        old_loss = new_loss


    projects = torch.tensor([])
    with torch.no_grad():
        for sample in tqdm(dataset):
            y = model([sample])
            projects = torch.cat([projects, y])

    reduce_dims_and_plot(projects,
                         y=None,
                         title=None,
                         file_name=None,
                         perplexity=50,
                         library='Multicore-TSNE',
                         perform_PCA=False,
                         projected=None,
                         figure_type='2d',
                         show_figure=True,
                         close_figure=False,
                         text=None)
