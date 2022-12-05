import numpy as np
from utils import *
from copy import copy
import gc
from pathlib import Path
import shutil
import warnings


class Hierarchical_Dirichlet_Point_Process:
    def __init__(self, num_neurons, particle_num, g_0, lambda_0, theta_0, g_noise, alpha, beta, normal_inverse_chi_squared_prior, bandwidth, refrence_time, threshold,
                 prune_threshold, merge_threshold, drop_threshold, window_size):

        self.num_neurons = num_neurons
        self.particle_num = particle_num
        self.g_0 = g_0
        self.lambda_0 = lambda_0
        self.theta_0 = theta_0
        self.alpha = alpha
        self.beta = beta
        self.g_noise = g_noise
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.prune_threshold = prune_threshold
        self.merge_threshold = merge_threshold
        self.drop_threshold = drop_threshold
        self.window_size = window_size
        self.refrence_time = refrence_time

        self.k_0 = normal_inverse_chi_squared_prior[0]
        self.v_0 = normal_inverse_chi_squared_prior[1]
        self.mu_0 = normal_inverse_chi_squared_prior[2]
        self.sigma2_0 = normal_inverse_chi_squared_prior[3]

        # create the directory to save sequences indicators and delete the previous one
        shutil.rmtree("./sequences", ignore_errors=True)
        Path("./sequences").mkdir(exist_ok=True)

        # sample background noise rate
        base_intensity = np.random.gamma(g_noise[0], 1.0/g_noise[1], self.particle_num)
        # create particles
        self.particles = []
        for i in range(self.particle_num):
            bkg_sequence = Sequence(index=1, num_neurons=num_neurons)
            particle = Particle(1.0 / self.particle_num, base_intensity[i])
            particle.sequences[1] = bkg_sequence
            self.particles.append(particle)

        self.sequence_g_0_distribution = np.zeros((num_neurons,), dtype=np.int32)

    def particle_filters(self, spike):
        # look back time duration
        tu = spike.spike_time - self.window_size
        self.active_interval = [tu, spike.spike_time]

        particles = []
        for i, particle in enumerate(self.particles):
            # sample sequence indicator and sequence type indicator
            particle, sequence_selected_index, seq_type_selected_index = self.sample_sequence_label(spike, particle, i)
            if sequence_selected_index != 1:
                # update parameters
                self.update_parameters(particle, spike, sequence_selected_index, seq_type_selected_index)
            else:
                # resample background noise rate
                particle.base_intensity = self.sample_base_intensity(particle)
            # calculate the value that used to update weight of a particle
            particle.log_prob = self.update_particle_log_prob(seq_type_selected_index, particle, spike)
            particles.append(particle)
        self.particles = particles

        # Particle normalization and resampling
        self.particles = self.particles_normalize_resampling(self.particles, self.threshold)

        if int(spike.index) % 100 == 0:
            gc.collect()

    def sample_sequence_label(self, spike, particle, particle_index):
        active_sequences_indexes = [0, 1]
        # for a new sequence
        sequence_rates = [self.g_0]
        log_likelihoods_dirichlet_categoricals = [
            log_dirichlet_categorical_distribution(self.sequence_g_0_distribution, spike.neuron_index, self.theta_0)]
        # for the sequence that contains background spikes:
        sequence_rates.append(particle.base_intensity)
        log_likelihoods_dirichlet_categoricals.append(
            log_dirichlet_categorical_distribution(self.sequence_g_0_distribution, spike.neuron_index, self.theta_0)
        )
        # update the active list of sequences
        particle.active_sequences = self.update_active_sequences(particle)
        # loop over current active sequences
        for active_sequence_index, _ in particle.active_sequences.items():
            active_sequences_indexes.append(active_sequence_index)
            # spikes in this sequence
            spikes_this_seq = particle.sequences[active_sequence_index].spikes
            central_time = particle.sequences[active_sequence_index].central_time
            # type indicator of this sequence
            seq_type_index_this_seq = particle.sequences_to_seq_type[active_sequence_index]
            if spike.neuron_index in spikes_this_seq:
                timeseq = np.array(spikes_this_seq[spike.neuron_index])[:, 0]
                # prune the sequence
                var = np.var(timeseq)
                if var > self.prune_threshold:
                    sum_timeseq = np.sum(timeseq)
                    spike_count = particle.sequences[active_sequence_index].spike_count
                    particle.sequences[active_sequence_index].central_time = (central_time * spike_count - sum_timeseq) / (spike_count - len(timeseq))
                    particle.sequences[active_sequence_index].spike_count -= len(timeseq)
                    particle.sequences[active_sequence_index].spike_distribution[spike.neuron_index] -= len(timeseq)
                    particle.sequences[1].spike_count += len(timeseq)
                    particle.sequences[1].spike_distribution += len(timeseq)
                    particle.seq_types[seq_type_index_this_seq].spike_count -= len(timeseq)
                    particle.seq_types[seq_type_index_this_seq].spike_distribution[spike.neuron_index] -= len(timeseq)
                    particle.sequences[active_sequence_index].amplitude = self.sample_seq_amplitude(particle.sequences[active_sequence_index])
                    del particle.sequences[active_sequence_index].spikes[spike.neuron_index]
                    time_intervals = spike.spike_time - particle.sequences[active_sequence_index].central_time
                else:
                    time_intervals = spike.spike_time - central_time
            else:
                time_intervals = spike.spike_time - central_time

            # calculate the prior
            seq_type_offset = particle.seq_types[seq_type_index_this_seq].offset_param[spike.neuron_index]
            seq_type_bias = particle.seq_types[seq_type_index_this_seq].bias_param[spike.neuron_index]
            seq_type_rate = particle.seq_types[seq_type_index_this_seq].rate_param[spike.neuron_index]
            time_intervals -= seq_type_offset
            amplitude_seq = particle.sequences[active_sequence_index].amplitude
            rate = triggering_gaussian_kernel(time_intervals, amplitude_seq*seq_type_rate, seq_type_bias)
            sequence_rates.append(rate)

            # calculate the likelihood
            log_likelihoods_dirichlet_categorical = log_dirichlet_categorical_distribution(
                particle.seq_types[seq_type_index_this_seq].spike_distribution, spike.neuron_index, self.theta_0
            )
            log_likelihoods_dirichlet_categoricals.append(log_likelihoods_dirichlet_categorical)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_sequence_rates = np.log(sequence_rates)
        sequence_prob = log_sequence_rates + log_likelihoods_dirichlet_categoricals
        # prevent overflow
        sequence_prob -= np.max(sequence_prob)
        sequence_prob = np.exp(sequence_prob)

        # sample sequence indicator
        sequence_draw = np.random.multinomial(1, pvals=sequence_prob / np.sum(sequence_prob))
        sequence_selected_index = np.array(active_sequences_indexes)[np.argmax(sequence_draw)]

        # create a new sequence
        if sequence_selected_index == 0:
            particle.num_sequences += 1
            sequence_selected_index = particle.num_sequences + 1
            sequence_selected = Sequence(sequence_selected_index, self.num_neurons)
            # update stats of this sequence
            sequence_selected.update_counts(spike)
            sequence_selected.central_time = spike.spike_time
            sequence_selected.spikes[spike.neuron_index] = [[spike.spike_time, spike.index]]
            particle.sequences[sequence_selected_index] = sequence_selected
            particle.active_sequences[sequence_selected_index] = []
        # background noise
        elif sequence_selected_index == 1:
            # update stats of this sequence
            particle.sequences[1].update_counts(spike)
            seq_type_selected_index = -1
        # reuse previous one
        else:
            sequence_selected = particle.sequences[sequence_selected_index]
            # update stats of this sequence
            sequence_selected.central_time = (
                sequence_selected.central_time * sequence_selected.spike_count + spike.spike_time) / (sequence_selected.spike_count + 1)
            sequence_selected.update_counts(spike)
            if spike.neuron_index in sequence_selected.spikes:
                sequence_selected.spikes[spike.neuron_index].append([spike.spike_time, spike.index])
            else:
                sequence_selected.spikes[spike.neuron_index] = [[spike.spike_time, spike.index]]

        if sequence_selected_index != 1:
            # merge similar sequences
            self.merge_sequences(sequence_selected_index, particle)
            # sample seq_type
            particle, seq_type_selected_index = self.sample_seq_type_label(particle, particle.sequences[sequence_selected_index], spike)
        else:
            seq_type_selected_index = -1

        return particle, sequence_selected_index, seq_type_selected_index

    def sample_seq_type_label(self, particle, sequence, spike=None):
        # for a new type
        seq_type_rates = [self.lambda_0]
        seq_type_log_likelihoods_dirichlet_categorical = [
            log_dirichlet_categorical_distribution(self.sequence_g_0_distribution, 0, self.theta_0)*sequence.spike_count
        ]
        seq_type_indexes = [0]
        # update active list of sequence types
        particle.active_seq_types = self.update_active_seq_types(particle, sequence)
        # remove current sequence from its type
        if sequence.index in particle.sequences_to_seq_type:
            old_seq_type_index = particle.sequences_to_seq_type[sequence.index]
            particle.seq_types[old_seq_type_index].remove_counts(sequence, spike=spike, exclude_cur=True)
        # loop over current types
        for seq_type_index, seq_type in list(particle.seq_types.items()):
            if len(particle.seq_type_to_sequences[seq_type_index]) == 0:
                del particle.seq_types[seq_type_index].spike_distribution, particle.seq_types[seq_type_index], particle.seq_type_to_sequences[seq_type_index]
                continue
            # calculate the prior

            if seq_type_index in particle.active_seq_types:
                timeseq = particle.active_seq_types[seq_type_index]
                seq_type_indexes.append(seq_type_index)
                time_intervals = sequence.central_time - timeseq
                seq_type_kernel_param = seq_type.kernel_param
                rate = triggering_seq_type_kernel(time_intervals, seq_type_kernel_param, self.bandwidth, self.refrence_time)
            else:
                continue
            seq_type_rates.append(rate)
            log_likelihoods_dirichlet_categorical_sum = 0

            # calculate the likelihood
            for spike_neuron_index, spike_times in sequence.spikes.items():
                log_likelihoods_dirichlet_categorical_sum += log_dirichlet_categorical_distribution(
                    seq_type.spike_distribution, spike_neuron_index, self.theta_0) * len(spike_times)
            seq_type_log_likelihoods_dirichlet_categorical.append(log_likelihoods_dirichlet_categorical_sum)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_seq_type_rates = np.log(seq_type_rates)

        seq_type_prob = log_seq_type_rates + seq_type_log_likelihoods_dirichlet_categorical
        # prevent overflow
        seq_type_prob -= np.max(seq_type_prob)
        seq_type_prob = np.exp(seq_type_prob)

        # sample type indicator
        seq_type_draw = np.random.multinomial(1, pvals=seq_type_prob / np.sum(seq_type_prob))
        seq_type_selected_index = np.array(seq_type_indexes)[np.argmax(seq_type_draw)]

        if sequence.index in particle.sequences_to_seq_type:
            old_seq_type_index = particle.sequences_to_seq_type[sequence.index]
            particle.seq_type_to_sequences[old_seq_type_index].remove(sequence.index)

        # create a new typr
        if seq_type_selected_index == 0:
            particle.num_seq_types += 1
            seq_type_selected_index = particle.num_seq_types + 1
            seq_type_selected = SeqType(seq_type_selected_index, self.num_neurons)
            particle.seq_types[seq_type_selected_index] = seq_type_selected
            particle.seq_type_to_sequences[seq_type_selected_index] = [sequence.index]
        # reuse previous one
        else:
            seq_type_selected = particle.seq_types[seq_type_selected_index]
            particle.seq_type_to_sequences[seq_type_selected_index].append(sequence.index)

        # update stats of this type
        seq_type_selected.update_counts(sequence)
        particle.sequences_to_seq_type[sequence.index] = seq_type_selected_index

        return particle, seq_type_selected_index

    def update_parameters(self, particle, spike, sequence_selected_index, seq_type_selected_index):
        selected_seq = particle.sequences[sequence_selected_index]
        selected_seq_type = particle.seq_types[seq_type_selected_index]
        if selected_seq_type.rate_param is None:
            selected_seq_type.rate_param = dirichlet(None, self.theta_0, None)
        else:
            selected_seq_type.rate_param = self.sample_seq_type_rate(selected_seq_type, particle)

        if selected_seq_type.offset_param is None or selected_seq_type.bias_param is None:
            offset, bias = normal_inverse_chi_squared(None, self.k_0, self.v_0, self.mu_0, self.sigma2_0, self.num_neurons)
            selected_seq_type.offset_param = offset
            selected_seq_type.bias_param = bias
        else:
            offset, bias = self.sample_seq_type_offset_and_bias(spike, selected_seq_type, particle)
            selected_seq_type.offset_param[spike.neuron_index] = offset
            selected_seq_type.bias_param[spike.neuron_index] = bias

        if selected_seq_type.kernel_param is None:
            selected_seq_type.kernel_param = np.random.gamma(self.alpha[0], 1.0 / (self.alpha[1]))
        else:
            selected_seq_type.kernel_param = self.sample_seq_type_kernel(selected_seq, seq_type_selected_index, particle)

        if selected_seq.amplitude is None:
            selected_seq.amplitude = np.random.gamma(self.beta[0], 1.0/self.beta[1])
        else:
            selected_seq.amplitude = self.sample_seq_amplitude(selected_seq)

    def sample_seq_type_rate(self, selected_seq_type, particle):
        # equation (5) in supplementary material
        spike_distribution = selected_seq_type.spike_distribution
        return dirichlet(None, self.theta_0 + spike_distribution, None)

    def sample_seq_type_offset_and_bias(self, spike, selected_seq_type, particle):
        seq_indexes = particle.seq_type_to_sequences[selected_seq_type.index]
        time_interval_sum = 0
        time_interval_square_sum = 0
        counts = 0
        # calculate z of equation (8) in supplementary material
        use_pre_spikes = False
        for seq_index in seq_indexes:
            central_time = particle.sequences[seq_index].central_time
            if hasattr(particle.sequences[seq_index], 'spikes') and len(particle.sequences[seq_index].spikes) != 0:
                spikes_this_seq = particle.sequences[seq_index].spikes
                if spike.neuron_index in spikes_this_seq:
                    intervals = np.array(spikes_this_seq[spike.neuron_index])[:, 0] - central_time
                    counts += len(intervals)
                    time_interval_sum += np.sum(intervals)
                    time_interval_square_sum += np.sum(intervals**2)
            else:
                use_pre_spikes = True
        # some sequences have been saved to disk, so load their stats
        if use_pre_spikes and spike.neuron_index in selected_seq_type.time_sum:
            time_interval_sum += selected_seq_type.time_sum[spike.neuron_index]
            time_interval_square_sum += selected_seq_type.time_sum_square[spike.neuron_index]

        # equation (8) in supplementary material
        n = selected_seq_type.spike_distribution[spike.neuron_index]
        k = self.k_0 + n
        v = self.v_0 + n
        mu = (time_interval_sum + self.k_0*self.mu_0)/k

        sigma2 = (
            self.v_0 * self.sigma2_0 +
            time_interval_square_sum - time_interval_sum * time_interval_sum / n +
            self.k_0 * n * (self.mu_0 - time_interval_sum/n) * (self.mu_0 - time_interval_sum/n) / k
        ) / v
        selected_seq_type.record_param = [k, v, mu, sigma2]
        return normal_inverse_chi_squared(None, k, v, mu, sigma2)

    def sample_seq_type_kernel(self, sequence, seq_type_selected_index, particle):
        # equation (6) in supplementary material
        sum_integral = 0
        timeseq = particle.active_seq_types[seq_type_selected_index]
        time_intervals = sequence.central_time - timeseq
        constant = np.exp(self.bandwidth*self.refrence_time)
        sum_integral = (len(timeseq)*constant - np.sum(np.exp(-1.0*self.bandwidth*np.abs(time_intervals)))) / self.bandwidth

        return np.random.gamma(self.alpha[0] + len(timeseq), 1.0 / (self.alpha[1] + sum_integral))

    def sample_seq_amplitude(self, sequence):
        # equation (9) in supplementary material
        spike_count = sequence.spike_count
        return np.random.gamma(self.beta[0] + spike_count, 1.0 / (self.beta[1] + 1))

    def merge_sequences(self, sequence_selected_index, particle):
        source_seq = particle.sequences[sequence_selected_index]
        central_time = source_seq.central_time
        # loop over current active sequences
        for seq_index in list(particle.active_sequences.keys()):
            if seq_index == sequence_selected_index:
                continue
            target_seq = particle.sequences[seq_index]
            # the times of two sequences are close
            if np.abs(central_time - target_seq.central_time) <= self.merge_threshold:
                source_seq.central_time = (source_seq.central_time*source_seq.spike_count + target_seq.central_time *
                                           target_seq.spike_count) / (source_seq.spike_count+target_seq.spike_count)
                source_seq.spike_count += target_seq.spike_count
                source_seq.spike_distribution += target_seq.spike_distribution
                source_seq.spikes = {key: source_seq.spikes.get(key, []) + target_seq.spikes.get(key, [])
                                     for key in set(list(source_seq.spikes.keys()) + list(target_seq.spikes.keys()))}
                source_seq.amplitude = np.random.gamma(self.beta[0] + source_seq.spike_count, 1.0 / (self.beta[1] + 1))

                # To add/remove target seq stats in seq_type
                if sequence_selected_index in particle.sequences_to_seq_type:
                    seq_type_selected_index = particle.sequences_to_seq_type[sequence_selected_index]
                    particle.seq_types[seq_type_selected_index].update_counts(target_seq)
                seq_type_index = particle.sequences_to_seq_type[seq_index]
                particle.seq_types[seq_type_index].remove_counts(target_seq)
                particle.seq_type_to_sequences[seq_type_index].remove(seq_index)
                if len(particle.seq_type_to_sequences[seq_type_index]) == 0:
                    del particle.seq_types[seq_type_index], particle.seq_type_to_sequences[seq_type_index]

                del target_seq.spikes, target_seq.spike_distribution, target_seq, particle.sequences_to_seq_type[
                    seq_index], particle.sequences[seq_index], particle.active_sequences[seq_index]

    def update_active_sequences(self, particle):
        tu = self.active_interval[0]
        # loop over current activate sequences
        for sequences_index in list(particle.active_sequences.keys()):
            central_time = np.mean(particle.sequences[sequences_index].central_time)
            # check whether this sequence is far away from current time
            if central_time <= tu:
                seq_type_index = particle.sequences_to_seq_type[sequences_index]
                # this sequence has few spikes after pruning, need to be dropped
                if particle.sequences[sequences_index].spike_count <= self.drop_threshold:
                    particle.sequences[1].spike_count += particle.sequences[sequences_index].spike_count
                    particle.sequences[1].spike_distribution += particle.sequences[sequences_index].spike_distribution
                    # To remove target seq stats in seq_type
                    particle.seq_types[seq_type_index].remove_counts(particle.sequences[sequences_index])
                    particle.seq_type_to_sequences[seq_type_index].remove(sequences_index)
                    if len(particle.seq_type_to_sequences[seq_type_index]) == 0:
                        del particle.seq_types[seq_type_index], particle.seq_type_to_sequences[seq_type_index]

                    del particle.sequences[sequences_index].spikes, particle.sequences[sequences_index].spike_distribution, particle.sequences[sequences_index], particle.sequences_to_seq_type[sequences_index]
                # for memory efficiency, save the sequence indicators of the spikes in this sequence to disk, and calculate some stats of this sequence
                else:
                    spikes_this_seq = particle.sequences[sequences_index].spikes
                    central_time = particle.sequences[sequences_index].central_time
                    seq_type_seq = particle.seq_types[seq_type_index]
                    for spike_neuron_index, timeseq in spikes_this_seq.items():
                        intervals = np.array(timeseq)[:, 0] - central_time
                        if spike_neuron_index not in seq_type_seq.time_sum:
                            seq_type_seq.time_sum[spike_neuron_index] = np.sum(intervals)
                            seq_type_seq.time_sum_square[spike_neuron_index] = np.sum(intervals**2)
                        else:
                            seq_type_seq.time_sum[spike_neuron_index] += np.sum(intervals)
                            seq_type_seq.time_sum_square[spike_neuron_index] += np.sum(intervals**2)

                    file_path = "./sequences/sequence_spikes_" + str(sequences_index)
                    np.save(file_path, particle.sequences[sequences_index].spikes)
                    del particle.sequences[sequences_index].spikes

                # remove this sequence from active list
                del particle.active_sequences[sequences_index]

        return particle.active_sequences

    def update_active_seq_types(self, particle, sequence):
        tu = self.active_interval[0]

        # update the active list of sequence types
        for seq_index, seq_seq_type_index in particle.sequences_to_seq_type.items():
            central_time = particle.sequences[seq_index].central_time
            if central_time <= tu - 2*self.window_size:
                # for memory efficiency, delete this
                if hasattr(particle.sequences[seq_index], 'spike_distribution'):
                    del particle.sequences[seq_index].spike_distribution
                # this sequence is much more far away from current time, so does not engage into the calculation of its sequence type's intensity
                continue
            if seq_index != sequence.index:
                if seq_seq_type_index in particle.active_seq_types:
                    particle.active_seq_types[seq_seq_type_index].append(central_time)
                else:
                    particle.active_seq_types[seq_seq_type_index] = [central_time]
        return particle.active_seq_types

    def final_drop(self):
        for particle_index, particle in enumerate(self.particles):
            for sequences_index in list(particle.sequences.keys()):
                if sequences_index != 1:
                    seq_type_index = particle.sequences_to_seq_type[sequences_index]
                    if particle.sequences[sequences_index].spike_count <= self.drop_threshold:
                        particle.sequences[1].spike_count += particle.sequences[sequences_index].spike_count
                        particle.sequences[1].spike_distribution += particle.sequences[sequences_index].spike_distribution

                        # To remove target seq stats in seq_type
                        particle.seq_types[seq_type_index].remove_counts(particle.sequences[sequences_index])
                        particle.seq_type_to_sequences[seq_type_index].remove(sequences_index)
                        if len(particle.seq_type_to_sequences[seq_type_index]) == 0:
                            del particle.seq_types[seq_type_index], particle.seq_type_to_sequences[seq_type_index]
                        del particle.sequences[sequences_index], particle.sequences_to_seq_type[sequences_index]

    def sample_base_intensity(self, particle):
        # equation (10) in supplementary material
        base_intensity = np.random.gamma(self.g_noise[0] + particle.sequences[1].spike_count - particle.sequences[1].pre_spike_count, 1.0 /
                                         (self.g_noise[1] + self.active_interval[1] - particle.sequences[1].pre_update_time))

        particle.sequences[1].pre_spike_count = particle.sequences[1].spike_count
        particle.sequences[1].pre_update_time = self.active_interval[1]
        return base_intensity

    def update_particle_log_prob(self, seq_type_selected_index, particle, spike):
        # equation (8) in paper
        if seq_type_selected_index == -1:
            log_likelihoods_dirichlet_categorical = log_dirichlet_categorical_distribution(self.sequence_g_0_distribution, spike.neuron_index, self.theta_0, exclude_cur=False)
        else:
            # remove current spike
            seq_type_spike_distribution = particle.seq_types[seq_type_selected_index].spike_distribution
            log_likelihoods_dirichlet_categorical = log_dirichlet_categorical_distribution(seq_type_spike_distribution, spike.neuron_index, self.theta_0, exclude_cur=True)
        return log_likelihoods_dirichlet_categorical

    def copy_particle(self, particle):
        # deepcopy is too slow to copy a particle
        new_particle = Particle(copy(particle.weight), copy(particle.base_intensity))
        new_particle.num_seq_types = copy(particle.num_seq_types)
        new_particle.num_sequences = copy(particle.num_sequences)
        new_particle.log_prob = copy(particle.log_prob)
        for index, seq_type in particle.seq_types.items():
            new_particle.seq_types[index] = seq_type.copy()

        for index, seq in particle.sequences.items():
            new_particle.sequences[index] = seq.copy()

        # for index, timeseq in particle.active_seq_types.items():
        #     new_particle.active_seq_types[index] = copy(timeseq)

        for index, timeseq in particle.active_sequences.items():
            new_particle.active_sequences[index] = copy(timeseq)

        for index, mo_index in particle.sequences_to_seq_type.items():
            new_particle.sequences_to_seq_type[index] = copy(mo_index)

        for index, seq_index in particle.seq_type_to_sequences.items():
            new_particle.seq_type_to_sequences[index] = copy(seq_index)

        return new_particle

    def particles_normalize_resampling(self, particles, threshold):
        weights = []
        log_probs = []
        for particle in particles:
            weights.append(particle.weight)
            log_probs.append(particle.log_prob)

        log_probs = np.array(log_probs)

        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)

        weights = np.array(weights)
        # equation (7) in paper
        weights *= probs

        # Normalize
        weights /= np.sum(weights)
        for i, particle in enumerate(particles):
            particle.weight = weights[i]

        # norm = np.sqrt(np.sum((weights - 1.0/self.particle_num)**2))
        norm = np.linalg.norm(weights)
        # print(norm)

        if norm > threshold + 1e-5:
            resample_num = self.particle_num
            # resample particles based on weights
            resample_draw = np.random.multinomial(resample_num, pvals=weights / np.sum(weights))

            new_particles = []
            for i, resample_times in enumerate(resample_draw):
                for _ in range(resample_times):
                    new_particles.append(self.copy_particle(particles[i]))

            # Reset weight
            for i, particle in enumerate(new_particles):
                particle.weight = 1.0 / self.particle_num
            return new_particles
        else:
            return particles
