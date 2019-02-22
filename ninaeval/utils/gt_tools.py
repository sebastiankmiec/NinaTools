from scipy.signal import butter, lfilter
from scipy.stats import multivariate_normal
import numpy as np

def get_start_end(orig_emg_data, orig_start, orig_end):
    """
        Corrects an initial estimate of the start and end range of a video window.

    :param orig_emg_data: All emg data (i.e. not simply orig_start to orig_end)
    :param orig_start: Index of start of video window (without buffer)
    :param orig_end: Index of end of video window (without buffer)

    :return: (int, int) start and end range of corrected movement window
    """

    num_emg_ch  = 16
    buffer      = 100 # to the start and end of the original range of data

    #
    # 1) Multi-variate version of Lidierth detection algorithm
    #   See: "Onset Detection in Surface Electromyographic Signals: A Systematic Comparison of Methods"
    #
    emg_start_indices   = [None for x in range(num_emg_ch)]
    emg_end_indices     = [None for x in range(num_emg_ch)]
    emg_data            = orig_emg_data[orig_start - buffer: orig_end + buffer]
    emg_data_copy       = np.copy(emg_data)                                         # for part (2)
    emg_data            = np.abs(emg_data)

    #
    # Compute Lidierth start/end index for each channel
    #
    for channel in range(num_emg_ch):

        # Apply sixth-order digital butterworth lowpass filter with 50 Hz cutoff frequency,
        fs          = 200
        nyquist     = 0.5 * fs
        cutoff      = 50
        order       = 6
        b, a        = butter(order, cutoff / nyquist, btype='lowpass')
        filt_data   = lfilter(b, a, emg_data[:, channel])


        # Configurable parameters
        window_size     = 50
        baseline_samp   = 200
        h               = 3     # Number of standard deviations to threshold with
        min_samples     = 90    # Must meet threshold this many times
        max_err         = 15    # Up to this many samples can fail to meet this threshold


        # States
        emg_start_idx   = None
        cur_idx         = window_size
        max_idx         = filt_data.shape[0]
        cur_avg         = np.mean(filt_data[0:window_size])
        cur_stddev      = np.std(filt_data[0:window_size])
        cur_count       = 0
        err_count       = 0


        #
        # Use test function to determine onset of signal (for current channel)
        #
        while (emg_start_idx is None) and (cur_idx <= max_idx):

            cur_test_func = (np.mean(filt_data[cur_idx-window_size:cur_idx]) - cur_avg) / cur_stddev

            # Possible onset
            if cur_test_func >= h:
                cur_count += 1

            else:
                if cur_count > 0:
                    err_count += 1

                # Too many errors for this potential onset
                if err_count >= max_err:
                    err_count = 0
                    cur_count = 0

            # Found an onset
            if cur_count >= min_samples:
                emg_start_idx = cur_idx - cur_count + 1

            # Update states
            cur_idx    += 1
            M           = min(baseline_samp, cur_idx)
            cur_avg     = np.mean(filt_data[0:M])
            cur_stddev  = np.std(filt_data[0:M])

        # Store onset index
        if not (emg_start_idx is None):
            emg_start_indices[channel] = emg_start_idx

        # States
        emg_end_idx     = None
        cur_idx         = max_idx-window_size
        cur_avg         = np.mean(filt_data[max_idx-window_size:max_idx])
        cur_stddev      = np.std(filt_data[max_idx-window_size:max_idx])
        cur_count       = 0
        err_count       = 0

        #
        # Use test function to determine end of signal (for current channel)
        #
        while (emg_end_idx is None) and (cur_idx >= 0):

            cur_test_func = (np.mean(filt_data[cur_idx:cur_idx + window_size]) - cur_avg) / cur_stddev

            # Possible onset
            if cur_test_func >= h:
                cur_count += 1

            else:
                if cur_count > 0:
                    err_count += 1

                # Too many errors for this potential onset
                if err_count >= max_err:
                    err_count = 0
                    cur_count = 0

            # Found an onset
            if cur_count >= min_samples:
                emg_end_idx = cur_idx + cur_count - 1

            # Update states
            cur_idx    -= 1
            M           = min(baseline_samp, max_idx-cur_idx)
            cur_avg     = np.mean(filt_data[max_idx-M:max_idx])
            cur_stddev  = np.std(filt_data[max_idx-M:max_idx])

        # Store end index
        if not (emg_end_idx is None):
            emg_end_indices[channel] = emg_end_idx


    # Pick the minimum (starting) index across channels
    valid_indices = [x for x in emg_start_indices if x is not None]
    if len(valid_indices) == 0:
        onset_idx = None
    else:
        onset_idx = min(valid_indices)

    # Pick the maximum (ending) index across channels
    valid_indices = [x for x in emg_end_indices if x is not None]
    if len(valid_indices) == 0:
        end_idx = None
    else:
        end_idx = max(valid_indices)


    #
    # 2) Generalized Likelihood Ratio Algorithm [using Lidierth as prior]
    #   See: "On the challenge of classifying 52 hand movements from surface electromyography"
    #
    emg_data        = emg_data_copy
    T               = emg_data.shape[0]
    min_start_idx   = 10
    min_interval    = int(np.ceil(0.3 * T))             # minimum interval length (end - start)

    max_likelikehood    = -np.inf
    max_indices         = None

    def compute_likelihood(t0, t1):
        """
            Compute the likelihood associated with parameters t0 and t1.
                > Store in max_likelihood and max_indices if relevant.

        :param t0: End of first rest period
        :param t1: End of movement and/or beginning of second rest period.
        """

        nonlocal max_likelikehood, max_indices, min_interval, emg_data, T
        if t1 - (t0 + min_interval) < min_interval:
            return

        #
        # Compute MLE estimate of rest distribution parameters
        #
        rest_one        = emg_data[0:t0]
        rest_two        = emg_data[t1:T]
        rest_samples    = np.concatenate((rest_one, rest_two), axis=0)
        rest_mean       = np.mean(rest_samples, axis=0)
        rest_cov        = np.cov(rest_samples, rowvar=False)
        rest_var        = np.diag(rest_cov)

        #
        # Compute MLE estimate of movement signal distribution parameters
        #
        sig_samples = emg_data[t0:t1]
        sig_mean    = np.mean(sig_samples, axis=0)
        sig_cov     = np.cov(sig_samples, rowvar=False)
        sig_var     = np.diag(sig_cov)

        # Can't have rest have more power than signal itself
        found_error = False
        for i in range(num_emg_ch):
            if rest_var[i] > sig_var[i]:
                found_error = True
                break
        if found_error:
            return

        try:
            rest_sum        = np.sum(np.log(multivariate_normal.pdf(x=rest_samples, mean=rest_mean, cov=rest_cov)))
            sig_sum         = np.sum(np.log(multivariate_normal.pdf(x=sig_samples, mean=sig_mean, cov=sig_cov)))
            log_likelihood  = rest_sum + sig_sum

            if log_likelihood > max_likelikehood:
                max_likelikehood    = log_likelihood
                max_indices         = (t0, t1)

        except np.linalg.LinAlgError as e:
            if 'singular matrix' in str(e):
                return
            else:
                raise (e)

    #
    # Use Lidierth estimate as a prior (if possible)
    #
    prior_window = 50

    if (onset_idx is None) and (end_idx is None):
        for t0 in range(min_start_idx, T-min_interval):
            for t1 in range(t0 + min_interval, T):
                compute_likelihood(t0, t1)

    elif (onset_idx is not None) and (end_idx is None):
        for t0 in range(max(onset_idx - prior_window, 0), min(onset_idx + prior_window, T)):
            for t1 in range(t0 + min_interval, T):
                compute_likelihood(t0, t1)

    elif (onset_idx is None) and (end_idx is not None):
        for t0 in range(min_start_idx, T-min_interval):
            for t1 in range(max(end_idx - prior_window, 0), min(end_idx + prior_window, T)):
                compute_likelihood(t0, t1)

    else:
        for t0 in range(max(onset_idx - prior_window, 0), min(onset_idx + prior_window, T)):
            for t1 in range(max(end_idx - prior_window, 0), min(end_idx + prior_window, T)):
                compute_likelihood(t0, t1)

    return max_indices