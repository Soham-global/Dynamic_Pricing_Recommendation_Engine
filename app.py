from flask import Flask, request, render_template
import joblib  # Changed from pickle to joblib
import numpy as np
from models.q_learning_agent import QLearningAgent
from models.pricing_environment import PricingEnvironment

application = Flask(__name__)
app = application

def load_object(file_path):
    """Load a model using joblib instead of pickle"""
    return joblib.load(file_path)

class PricingRecommendation:
    """Handle price recommendation logic"""
    
    def __init__(self):
        self.demand_model = None
        self.rl_agent = None
        self.pricing_env = None
        self.encoders = None
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Updated file extensions to .joblib
            self.demand_model = load_object('models/best_demand_model.joblib')
            self.rl_agent = load_object('models/rl_agent.joblib')
            self.pricing_env = load_object('models/pricing_environment.joblib')
            self.encoders = load_object('models/encoders.joblib')
            print("✅ All models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise e
    
    def encode_features(self, inventory_level, customer_segment, category):
        """Encode categorical features"""
        inventory_encoded = self.encoders['inventory_encoder'].transform([inventory_level])[0]
        segment_encoded = self.encoders['segment_encoder'].transform([customer_segment])[0]
        category_encoded = self.encoders['category_encoder'].transform([category])[0]
        return inventory_encoded, segment_encoded, category_encoded
    
    def engineer_features(self, current_price, competitor_price, cost):
        """Create engineered features"""
        price_difference = competitor_price - current_price
        discount_percentage = ((competitor_price - current_price) / competitor_price * 100) if competitor_price > 0 else 0
        price_ratio = current_price / competitor_price if competitor_price > 0 else 1
        profit_margin = ((current_price - cost) / current_price * 100) if current_price > 0 else 0
        rolling_avg_demand = 45  # Default value for new data
        return price_difference, discount_percentage, price_ratio, profit_margin, rolling_avg_demand
    
    def get_recommendation(self, input_data):
        """Get price recommendation"""
        try:
            # Extract input data
            current_price = float(input_data['current_price'])
            competitor_price = float(input_data['competitor_price'])
            cost = float(input_data['cost'])
            day_of_week = int(input_data['day_of_week'])
            month = int(input_data['month'])
            is_weekend = int(input_data['is_weekend'])
            is_festival = int(input_data['is_festival'])
            inventory_level = input_data['inventory_level']
            customer_segment = input_data['customer_segment']
            category = input_data['category']
            
            # Encode categorical features
            inventory_encoded, segment_encoded, category_encoded = self.encode_features(
                inventory_level, customer_segment, category
            )
            
            # Engineer features
            price_diff, discount_pct, price_ratio, profit_margin, rolling_avg = self.engineer_features(
                current_price, competitor_price, cost
            )
            
            # Create features dictionary
            features_dict = {
                'current_price': current_price,
                'competitor_price': competitor_price,
                'cost': cost,
                'price_difference': price_diff,
                'discount_percentage': discount_pct,
                'price_ratio': price_ratio,
                'profit_margin': profit_margin,
                'day_of_week': day_of_week,
                'month': month,
                'is_weekend': is_weekend,
                'is_festival': is_festival,
                'inventory_level_encoded': inventory_encoded,
                'customer_segment_encoded': segment_encoded,
                'category_encoded': category_encoded,
                'rolling_avg_demand': rolling_avg
            }
            
            # Create features array
            features_array = [
                current_price, competitor_price, cost,
                price_diff, discount_pct, price_ratio, profit_margin,
                day_of_week, month, is_weekend, is_festival,
                inventory_encoded, segment_encoded, category_encoded,
                rolling_avg
            ]
            
            # Get state
            state = self.pricing_env.discretize_state(features_dict)
            
            # Get best action from RL agent
            best_action = self.rl_agent.get_action(state, training=False)
            
            # Apply action and get recommendation
            recommended_price, predicted_demand, expected_profit = self.pricing_env.apply_action(
                features_array, best_action
            )
            
            # Calculate current metrics
            current_demand = self.demand_model.predict([features_array])[0]
            current_profit = (current_price - cost) * current_demand
            
            # Calculate improvement
            profit_improvement = ((expected_profit - current_profit) / current_profit * 100) if current_profit > 0 else 0
            
            results = {
                'current_price': f"₹{current_price:,.2f}",
                'recommended_price': f"₹{recommended_price:,.2f}",
                'action': self.pricing_env.action_names[best_action],
                'current_demand': f"{current_demand:.0f} units",
                'predicted_demand': f"{predicted_demand:.0f} units",
                'current_profit': f"₹{current_profit:,.2f}",
                'expected_profit': f"₹{expected_profit:,.2f}",
                'improvement_pct': f"{profit_improvement:+.2f}%",
                'improvement_class': 'positive' if profit_improvement > 0 else 'negative'
            }
            
            print(f"✅ Recommendation generated: {results['action']}")
            return results
            
        except Exception as e:
            print(f"❌ Error in recommendation: {e}")
            return None

# Initialize pricing recommendation system
print("🔄 Loading models...")
pricing_system = PricingRecommendation()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Handle price prediction"""
    if request.method == 'GET':
        return render_template('home.html', results=None)
    else:
        try:
            # Get form data
            input_data = {
                'current_price': request.form.get('current_price'),
                'competitor_price': request.form.get('competitor_price'),
                'cost': request.form.get('cost'),
                'day_of_week': request.form.get('day_of_week'),
                'month': request.form.get('month'),
                'is_weekend': request.form.get('is_weekend'),
                'is_festival': request.form.get('is_festival'),
                'inventory_level': request.form.get('inventory_level'),
                'customer_segment': request.form.get('customer_segment'),
                'category': request.form.get('category')
            }
            
            # Get recommendation
            results = pricing_system.get_recommendation(input_data)
            
            if results:
                return render_template('home.html', results=results)
            else:
                return render_template('home.html', results=None, error="Error generating recommendation")
            
        except Exception as e:
            print(f"❌ Error in prediction: {str(e)}")
            return render_template('home.html', results=None, error=str(e))

if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    print("📍 Open browser at: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
