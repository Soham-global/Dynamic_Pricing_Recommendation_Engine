class PricingEnvironment:
    """
    Reinforcement Learning Environment for Dynamic Pricing
    """
    def __init__(self, demand_model, feature_columns):
        self.demand_model = demand_model
        self.feature_columns = feature_columns
        
        # Define 5 pricing actions
        self.actions = {
            0: -0.10,  # Decrease price by 10%
            1: -0.05,  # Decrease price by 5%
            2: 0.00,   # Keep price same
            3: 0.05,   # Increase price by 5%
            4: 0.10    # Increase price by 10%
        }
        
        self.action_names = {
            0: 'Decrease 10%',
            1: 'Decrease 5%',
            2: 'Keep Same',
            3: 'Increase 5%',
            4: 'Increase 10%'
        }
        
        self.n_actions = len(self.actions)
    
    def discretize_state(self, features_dict):
        """
        Convert continuous features to discrete state
        """
        # Extract key features for state
        price_diff = features_dict['price_difference']
        is_peak = features_dict['is_festival'] or features_dict['is_weekend']
        inventory = features_dict['inventory_level_encoded']
        
        # Discretize price difference
        if price_diff < -200:
            price_state = 0  # Much more expensive
        elif price_diff < -50:
            price_state = 1  # Expensive
        elif price_diff < 50:
            price_state = 2  # Similar price
        elif price_diff < 200:
            price_state = 3  # Cheaper
        else:
            price_state = 4  # Much cheaper
        
        # Peak time: 0 or 1
        peak_state = 1 if is_peak else 0
        
        # Inventory: 0, 1, 2 (Low, Medium, High)
        inv_state = int(inventory)
        
        # Combine into single state number
        state = price_state * 6 + peak_state * 3 + inv_state
        
        return min(state, 29)  # Ensure state is within bounds (0-29)
    
    def predict_demand(self, features_array):
        """
        Predict demand using the trained ML model
        """
        demand = self.demand_model.predict([features_array])[0]
        return max(0, demand)  # Demand cannot be negative
    
    def calculate_reward(self, price, cost, demand):
        """
        Calculate profit (reward)
        """
        profit = (price - cost) * demand
        return profit
    
    def apply_action(self, current_features, action):
        """
        Apply price adjustment and return new state, reward
        """
        # Get current values
        current_price = current_features[0]  # current_price
        competitor_price = current_features[1]  # competitor_price
        cost = current_features[2]  # cost
        
        # Apply price adjustment
        price_change = self.actions[action]
        new_price = current_price * (1 + price_change)
        
        # Ensure price doesn't go below cost
        new_price = max(new_price, cost * 1.1)  # At least 10% margin
        
        # Create new feature array with updated price
        new_features = current_features.copy()
        new_features[0] = new_price
        
        # Update price-dependent features
        new_features[3] = competitor_price - new_price  # price_difference
        new_features[4] = ((competitor_price - new_price) / competitor_price * 100)  # discount_percentage
        new_features[5] = new_price / competitor_price  # price_ratio
        new_features[6] = ((new_price - cost) / new_price * 100)  # profit_margin
        
        # Predict demand with new price
        predicted_demand = self.predict_demand(new_features)
        
        # Calculate reward (profit)
        reward = self.calculate_reward(new_price, cost, predicted_demand)
        
        return new_price, predicted_demand, reward