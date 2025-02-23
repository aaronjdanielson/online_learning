import numpy as np
import pandas as pd

class LaplaceApproximateFilter:
    """
    A Laplace Approximation-based filter for state-space models with binomial observations.
   
    The model considers a sequence of binomial outcomes:
        y_t ~ Binomial(n_t, sigma(theta_t))
    with the latent state evolving as:
        theta_t = theta_{t-1} + epsilon_t,
    where epsilon_t ~ N(0, omega_t^2) and
        omega_t^2 = (tau(t) - tau(t-1)) * sigma_theta^2.
   
    The posterior update at time t is approximated via a second-order Taylor (Laplace) expansion
    around the prior mean (theta_{t-1}). The update equations are:
   
        theta_t = theta_{t-1} + (omega_t^2 * g_t) / (1 - omega_t^2 * h_t)
        sigma_t^2 = omega_t^2 / (1 - omega_t^2 * h_t)
   
    where:
        g_t = y_t - n_t * sigma(theta_{t-1})
        h_t = -n_t * sigma(theta_{t-1}) * (1 - sigma(theta_{t-1}))
    and sigma(theta) is the sigmoid function.
   
    Parameters
    ----------
    initial_theta : float
        The initial value for the latent state theta (i.e. theta_0).
    sigma_theta : float
        The standard deviation per unit time for the state noise.
        (State noise variance per day will be sigma_theta^2 * delta, where delta is the time difference in days.)
    initial_sigma2 : float, optional (default=1.0)
        The initial variance of the latent state.
    date_col : str, optional (default="date")
        The name of the date column in the input DataFrame.
    y_col : str, optional (default="y")
        The name of the column containing the number of successes.
    n_col : str, optional (default="n")
        The name of the column containing the number of trials.
    """

    def __init__(self, initial_theta: float, sigma_theta: float, initial_sigma2: float = 1.0,
                 date_col: str = "date", y_col: str = "y", n_col: str = "n"):
        self.initial_theta = initial_theta
        self.sigma_theta = sigma_theta
        self.initial_sigma2 = initial_sigma2
        self.date_col = date_col
        self.y_col = y_col
        self.n_col = n_col

    @staticmethod
    def _sigmoid(theta: float) -> float:
        """Compute the sigmoid function."""
        return 1.0 / (1.0 + np.exp(-theta))
   
    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the approximate Kalman filter with Laplace approximation to the input DataFrame.
       
        The DataFrame is assumed to have columns corresponding to the y, n, and date.
        The method adds two columns:
            - 'theta_post': the posterior mean estimate of theta at each time step.
            - 'sigma2_post': the posterior variance estimate of theta at each time step.
       
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with columns for observations ('y'), number of trials ('n'),
            and time stamps ('date').
           
        Returns
        -------
        pd.DataFrame
            DataFrame with additional columns 'theta_post' and 'sigma2_post'.
        """
        # Ensure the date column is of datetime type and sort by date
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(by=self.date_col).reset_index(drop=True)
       
        # Initialize columns for posterior mean and variance
        theta_post = np.zeros(len(df))
        sigma2_post = np.zeros(len(df))
       
        # Set initial state: For the first observation, use the prior
        theta_post[0] = self.initial_theta
        sigma2_post[0] = self.initial_sigma2
       
        # Iterate over the DataFrame (starting from the second observation)
        for t in range(1, len(df)):
            # Compute the time difference (delta) in days
            current_date = df.loc[t, self.date_col]
            previous_date = df.loc[t-1, self.date_col]
            delta_days = (current_date - previous_date).total_seconds() / (3600 * 24)
           
            # Compute the state noise variance for this step: omega_t^2
            omega2_t = delta_days * (self.sigma_theta ** 2)
           
            # Retrieve previous posterior state (theta_{t-1})
            theta_prev = theta_post[t-1]
           
            # Retrieve current observation data
            y_t = df.loc[t, self.y_col]
            n_t = df.loc[t, self.n_col]
           
            # Compute the sigmoid at the previous state: sigma(theta_{t-1})
            sigma_theta_prev = self._sigmoid(theta_prev)
           
            # Compute derivatives of the log-likelihood at theta_{t-1}
            g_t = y_t - n_t * sigma_theta_prev
            h_t = -n_t * sigma_theta_prev * (1 - sigma_theta_prev)
           
            # Compute the posterior mean (mode) update using the Laplace approximation:
            # theta_t = theta_{t-1} + (omega_t^2 * g_t) / (1 - omega_t^2 * h_t)
            denominator = 1 - omega2_t * h_t
            if denominator == 0:
                raise ZeroDivisionError(f"Encountered zero denominator at time index {t}.")
            theta_update = theta_prev + (omega2_t * g_t) / denominator
           
            # Compute the posterior variance:
            # sigma_t^2 = omega_t^2 / (1 - omega_t^2 * h_t)
            sigma2_update = omega2_t / denominator
           
            # Store updated values
            theta_post[t] = theta_update
            sigma2_post[t] = sigma2_update
       
        # Add posterior estimates as new columns in the DataFrame
        df['theta_post'] = theta_post
        df['sigma2_post'] = sigma2_post
       
        return df

# Example usage:
if __name__ == "__main__":
    # Create a sample DataFrame
    data = {
        "date": ["2025-01-01", "2025-01-02", "2025-01-04", "2025-01-07"],
        "y": [5, 7, 6, 8],
        "n": [10, 10, 10, 10]
    }
    df_sample = pd.DataFrame(data)
   
    # Initialize the filter with chosen hyperparameters
    initial_theta = 0.0
    sigma_theta = 0.5  # state noise per day
    filter_model = LaplaceApproximateFilter(initial_theta, sigma_theta, initial_sigma2=1.0)
   
    # Apply the filter to the DataFrame
    df_filtered = filter_model.filter_dataframe(df_sample)
    print(df_filtered)