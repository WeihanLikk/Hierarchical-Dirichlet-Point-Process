from copy import deepcopy, copy
import numpy as np


class Particle:
    def __init__(self, weight, base_intensity) -> None:
        self.weight = weight
        self.active_sequences = {}
        self.active_seq_types = {}
        self.sequences = {}
        self.seq_types = {}
        self.sequences_to_seq_type = {}
        self.seq_type_to_sequences = {}
        self.num_sequences = 0
        self.num_seq_types = 0
        self.log_prob = 0
        self.base_intensity = base_intensity


class Sequence:
    def __init__(self, index, num_neurons) -> None:
        self.index = index
        self.num_neurons = num_neurons
        self.amplitude = None
        self.spike_distribution = np.zeros((num_neurons,), dtype=np.int32)
        self.spike_count = 0
        self.central_time = 0
        self.pre_spike_count = 0
        self.pre_update_time = 0
        self.spikes = {}

    def update_counts(self, spike):
        self.spike_distribution[spike.neuron_index] += 1
        self.spike_count += 1

    def copy(self):
        new_seq = Sequence(self.index, self.num_neurons)
        new_seq.amplitude = copy(self.amplitude)
        new_seq.spike_count = copy(self.spike_count)
        new_seq.central_time = copy(self.central_time)
        new_seq.pre_spike_count = copy(self.pre_spike_count)
        new_seq.pre_update_time = copy(self.pre_update_time)
        if hasattr(self, 'spike_distribution'):
            new_seq.spike_distribution = copy(self.spike_distribution)
        if hasattr(self, 'spikes'):
            for spike_neuron, timseq in self.spikes.items():
                new_seq.spikes[spike_neuron] = copy(timseq)
        return new_seq


class SeqType:
    def __init__(self, index, num_neurons) -> None:
        self.index = index
        self.kernel_param = None
        self.rate_param = None
        self.offset_param = None
        self.bias_param = None
        self.num_neurons = num_neurons
        self.spike_distribution = None
        self.spike_count = 0
        self.record_param = []
        self.time_sum = {}
        self.time_sum_square = {}

    def update_counts_from_spike(self, spike):
        self.spike_distribution[spike.neuron_index] += 1
        self.spike_count += 1

    def update_counts(self, sequences):
        if self.spike_distribution is None:
            self.spike_distribution = deepcopy(sequences.spike_distribution)
        else:
            self.spike_distribution += sequences.spike_distribution
        self.spike_count += sequences.spike_count

    def remove_counts(self, sequence, spike=None, exclude_cur=False):
        if exclude_cur:
            spike_count = sequence.spike_count - 1
            self.spike_distribution -= sequence.spike_distribution
            self.spike_distribution[spike.neuron_index] += 1
            self.spike_count -= spike_count
        else:
            self.spike_distribution -= sequence.spike_distribution
            self.spike_count -= sequence.spike_count

    def reset_counts(self):
        if self.spike_distribution is not None:
            del self.spike_distribution
        self.spike_distribution = np.zeros((self.num_neurons,), dtype=np.int32)
        self.spike_count = 0

    def copy(self):
        new_seq_type = SeqType(self.index, self.num_neurons)
        new_seq_type.kernel_param = copy(self.kernel_param)
        new_seq_type.rate_param = copy(self.rate_param)
        new_seq_type.offset_param = copy(self.offset_param)
        new_seq_type.bias_param = copy(self.bias_param)
        new_seq_type.spike_distribution = copy(self.spike_distribution)
        new_seq_type.spike_count = copy(self.spike_count)
        new_seq_type.record_param = copy(self.record_param)

        for spike_neuron, sum in self.time_sum.items():
            new_seq_type.time_sum[spike_neuron] = copy(sum)

        for spike_neuron, sum_square in self.time_sum_square.items():
            new_seq_type.time_sum_square[spike_neuron] = copy(sum_square)

        return new_seq_type


class Spike:
    def __init__(self, index, neuron_index, spike_time, num_neurons) -> None:
        self.index = index
        self.neuron_index = neuron_index
        self.spike_time = spike_time


def log_dirichlet_categorical_distribution(seq_spike_distribution, spike_index, priors, exclude_cur=False):
    exclude_cur_spike = 1 if exclude_cur else 0
    prob = (priors[spike_index] + seq_spike_distribution[spike_index] - exclude_cur_spike) / (np.sum(priors) + np.sum(seq_spike_distribution)-exclude_cur_spike)
    log_prob = np.log(prob)
    return log_prob


def triggering_seq_type_kernel(time_interval, kernel_param, bandwith, refrence_time):
    return np.sum(kernel_param * np.exp(-1.0 * bandwith * (np.abs(time_interval) - refrence_time)))


def triggering_gaussian_kernel(time_interval, kernel_param, bias):
    numerator = -1.0 * time_interval*time_interval / (2 * bias)
    denominator = (2 * bias * np.pi)**0.5
    return kernel_param * np.exp(numerator) / denominator


def normal_inverse_chi_squared(rng, k, v, mu, sigma2, size=None):

    bias = v*sigma2 / np.random.chisquare(v, size)
    offset = np.random.normal(mu, bias/k, size)

    return offset, bias


def dirichlet(rng, prior, size):

    return np.random.dirichlet(prior, size)


def create_mask(rng, num_neurons, max_time, mask_length, percent_masked):
    masked_area = percent_masked * max_time * num_neurons / 100
    num_masked = int(np.ceil(masked_area / mask_length))

    intervals = []
    for t in np.arange(0, max_time - mask_length, mask_length):
        intervals.append([t, t+mask_length])

    time_interval_len = len(intervals)

    mask_intervals = np.array(list(range(0, num_neurons*time_interval_len)))
    mask_indexes = rng.choice(mask_intervals, num_masked, replace=False)
    mask = np.ones((num_neurons*time_interval_len,))
    mask[mask_indexes] = 0
    mask = np.reshape(mask, (num_neurons, time_interval_len))

    spike_mask = {}
    for n in range(num_neurons):
        mask_indexes_this_neuron = np.where(mask[n, :] == 0)[0].tolist()
        start_times = []
        end_times = []
        for index in mask_indexes_this_neuron:
            start_times.append(intervals[index][0])
            end_times.append(intervals[index][1])

        indexes_to_delete = []
        for index in range(len(start_times)-1):
            if start_times[index+1] - start_times[index] < 1e-4:
                indexes_to_delete.append(index)

        num_deleted = 0
        for index in range(len(start_times)):
            if index in indexes_to_delete:
                num_deleted += 1
            else:
                start_time = start_times[index - num_deleted]
                end_time = end_times[index]
                num_deleted = 0
                if n not in spike_mask:
                    spike_mask[n] = [[start_time, end_time]]
                else:
                    spike_mask[n].append([start_time, end_time])

    return spike_mask


def apply_mask(spikes, mask):
    masked_spikes = []
    unmasked_spikes = []

    for i in range(len(spikes)):
        within_mask = False
        neuron_index = spikes[i, 0]
        spike_time = spikes[i, 1]
        if neuron_index in mask:
            masked_times = mask[neuron_index]
            for start, end in masked_times:
                if start < spike_time and spike_time < end:
                    within_mask = True
                    masked_spikes.append([neuron_index, spike_time])
                    break
        if not within_mask:
            unmasked_spikes.append([neuron_index, spike_time])
    unmasked_spikes = np.array(unmasked_spikes)
    masked_spikes = np.array(masked_spikes)
    return unmasked_spikes, masked_spikes

def sort_neurons(spikes, particle, threshold):
    seq_type_indexes = []
    for seq_type_index, seq_type in particle.seq_types.items():
        if len(particle.seq_type_to_sequences[seq_type_index]) != 0:
            seq_type_indexes.append(seq_type_index)
    seq_type_indexes = np.array(seq_type_indexes)
    rates = []
    offsets = []
    for seq_type_index in seq_type_indexes:
        rates.append(particle.seq_types[seq_type_index].rate_param)
        offsets.append(particle.seq_types[seq_type_index].offset_param)
    rates = np.array(rates)
    offsets = np.array(offsets)
    max_rates_among_seq_types = np.max(rates, axis=0)
    rate_threshold = threshold
    active_by_rate = max_rates_among_seq_types >= np.quantile(max_rates_among_seq_types, rate_threshold)
    active_by_rate = active_by_rate * 1
    preferred_type = np.argmax(rates, axis=0).tolist()
    preferred_delay = [offsets[r, n] for (n, r) in enumerate(preferred_type)]
    info = list(zip(active_by_rate.tolist(), preferred_type, preferred_delay))
    sorted_neuron_indexes = sorted(range(len(info)), key=info.__getitem__, reverse=False)

    sorted_spikes = np.zeros_like(spikes)
    for i in range(len(sorted_neuron_indexes)):
        target_spikes = spikes[np.where(spikes[:, 0] == sorted_neuron_indexes[i]+1)]
        target_spikes[:, 0] = i+1
        sorted_spikes[np.where(spikes[:, 0] == sorted_neuron_indexes[i]+1)] = target_spikes
    return sorted_spikes
