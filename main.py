import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

# Constants
TREATMENTS = [0, 250, 500, 750]
NUM_TREATMENTS = len(TREATMENTS)
BUDGET_LIMIT = 5_000_000
NUM_CUSTOMERS = 1000  # Example number of customers

# Define customer segments
SEGMENTS = ['young_professionals', 'mid_parents', 'early_retirees', 'golden_age']
NUM_SEGMENTS = len(SEGMENTS)

# Thompson Sampling parameters
ALPHA = 1
BETA = 1

# Initialize Beta distribution parameters for each treatment and segment
alpha_params = np.ones((NUM_SEGMENTS, NUM_TREATMENTS))
beta_params = np.ones((NUM_SEGMENTS, NUM_TREATMENTS))

# Initialize logging
log_data = []

# Create a directory for saving plots if it doesn't exist
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Reward function based on customer segment
def get_ltv(treatment, customer_id):
    # Placeholder function to simulate LTV
    # Replace with real-world data or a sophisticated model
    segment = customer_segments[customer_id]
    return np.random.uniform(-100, 500)  # Simulate LTV between -100 and 500

def get_reward(treatment, customer_id):
    ltv = get_ltv(treatment, customer_id)
    return 1 if ltv >= 0 else 0  # Reward is 1 if LTV is non-negative, otherwise 0

# Create customer segments
def assign_customer_segment(customer_id):
    # Randomly assign a segment for demonstration purposes
    # Replace with real segmentation logic
    return random.choice(SEGMENTS)

customer_segments = [assign_customer_segment(i) for i in range(NUM_CUSTOMERS)]

# Train the Thompson Sampling model
for customer_id in range(NUM_CUSTOMERS):
    # Determine customer segment
    segment = customer_segments[customer_id]
    segment_index = SEGMENTS.index(segment)
    
    # Sample from the Beta distributions for each treatment
    samples = np.random.beta(alpha_params[segment_index], beta_params[segment_index])
    
    # Choose the treatment with the highest sampled value
    treatment_index = np.argmax(samples)
    treatment = TREATMENTS[treatment_index]
    
    # Simulate reward for the treatment
    reward = get_reward(treatment, customer_id)
    
    # Log the details
    log_data.append({
        'customer_id': customer_id,
        'segment': segment,
        'treatment': treatment,
        'reward': reward,
        'sampled_values': samples
    })
    
    # Update Beta distribution parameters based on the reward
    if reward == 1:
        alpha_params[segment_index, treatment_index] += 1
    else:
        beta_params[segment_index, treatment_index] += 1
    
    # Plot Beta distributions
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 1, 100)
    for i in range(NUM_TREATMENTS):
        y = np.random.beta(alpha_params[segment_index, i], beta_params[segment_index, i], 1000)
        plt.hist(y, bins=30, alpha=0.5, label=f'Treatment £{TREATMENTS[i]} (α={alpha_params[segment_index, i]}, β={beta_params[segment_index, i]})')
    
    plt.title(f'Beta Distributions for Segment {segment} After Customer {customer_id}')
    plt.xlabel('Success Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot as an image
    plt.savefig(f'plots/beta_distributions_{segment}_{customer_id}.png')
    plt.close()

    # Create interactive plot using Plotly
    fig = go.Figure()
    for i in range(NUM_TREATMENTS):
        x = np.linspace(0, 1, 100)
        y = np.random.beta(alpha_params[segment_index, i], beta_params[segment_index, i], 1000)
        fig.add_trace(go.Histogram(x=y, histnorm='probability', name=f'Treatment £{TREATMENTS[i]} (α={alpha_params[segment_index, i]}, β={beta_params[segment_index, i]})'))

    fig.update_layout(
        title=f'Beta Distributions for Segment {segment} After Customer {customer_id}',
        xaxis_title='Success Probability',
        yaxis_title='Probability',
        barmode='overlay'
    )
    
    # Save the interactive plot as an HTML file
    pio.write_html(fig, file=f'plots/beta_distributions_{segment}_{customer_id}.html')

# Save log data to a CSV file
log_df = pd.DataFrame(log_data)
log_df.to_csv('thompson_sampling_log.csv', index=False)

# Determine recommended treatments for each customer
def recommend_treatment_for_customer(customer_id):
    segment = customer_segments[customer_id]
    segment_index = SEGMENTS.index(segment)
    
    # Sample from the Beta distributions to determine the best treatment
    samples = np.random.beta(alpha_params[segment_index], beta_params[segment_index])
    treatment_index = np.argmax(samples)
    return TREATMENTS[treatment_index]

# Allocate treatments considering budget constraint
def allocate_treatments(customers):
    total_spent = 0
    treatment_allocation = {}
    
    for customer_id in customers:
        treatment = recommend_treatment_for_customer(customer_id)
        ltv = get_ltv(treatment, customer_id)
        if ltv >= 0 and total_spent + treatment <= BUDGET_LIMIT:
            treatment_allocation[customer_id] = treatment
            total_spent += treatment
        else:
            treatment_allocation[customer_id] = 0  # Default to no treatment if LTV is negative or budget is exceeded

    return treatment_allocation

# Example usage
customers = range(NUM_CUSTOMERS)
allocation = allocate_treatments(customers)

# Print the treatment allocation
print(f"Treatment Allocation: {allocation}")

# Print the number of treatments assigned and budget used
total_treatment_cost = sum(allocation.values())
print(f"Total treatment cost: £{total_treatment_cost}")
print(f"Remaining budget: £{BUDGET_LIMIT - total_treatment_cost}")

