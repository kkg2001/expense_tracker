import streamlit as st
import pandas as pd
import numpy as np
import json
import uuid
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

# Set page config
st.set_page_config(
    page_title="Expense Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --info-color: #17a2b8;
        --light-bg: #f8f9fa;
        --dark-bg: #343a40;
    }
    
    /* Custom styling for metrics */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Enhanced sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom card styling */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Form styling */
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        color: #667eea;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Category icons */
    .category-icon {
        display: inline-block;
        margin-right: 0.5rem;
    }
    
    /* Success/Error alerts styling */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Table styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class ExpenseTracker:
    def __init__(self):
        self.data_file = "expenses.json"
        self.load_data()
    
    def load_data(self):
        """Load expense data from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # Empty or invalid file: start with no expenses
                    data = {'expenses': []}
                self.expenses = data.get('expenses', [])
        except FileNotFoundError:
            self.expenses = []
    
    def save_data(self):
        """Save expense data to JSON file"""
        data = {'expenses': self.expenses}
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def add_expense(self, amount: float, description: str, category: str = None, date: str = None):
        """Add a new expense with categorization"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        if category is None:
            category = self.categorize_expense(description)
        
        expense = {
            'id': str(uuid.uuid4()),
            'date': date,
            'amount': amount,
            'description': description,
            'category': category,
            'created_at': datetime.now().isoformat()
        }
        
        self.expenses.append(expense)
        self.save_data()
        return expense
    
    def categorize_expense(self, description: str) -> str:
        """Expense categorization based on keywords"""
        description_lower = description.lower()
        
        # Define category keywords
        categories = {
            'Food & Dining': ['restaurant', 'food', 'meal', 'lunch', 'dinner', 'breakfast', 'cafe', 'pizza', 'burger', 'groceries', 'supermarket'],
            'Transportation': ['gas', 'fuel', 'taxi', 'uber', 'lyft', 'bus', 'train', 'parking', 'car', 'metro'],
            'Shopping': ['amazon', 'store', 'mall', 'shopping', 'clothes', 'electronics', 'shoes', 'purchase'],
            'Entertainment': ['movie', 'cinema', 'game', 'concert', 'show', 'netflix', 'spotify', 'entertainment'],
            'Health & Medical': ['doctor', 'hospital', 'pharmacy', 'medicine', 'health', 'dental', 'medical'],
            'Utilities': ['electricity', 'water', 'gas bill', 'internet', 'phone', 'utility', 'bill'],
            'Education': ['book', 'course', 'school', 'university', 'education', 'tuition', 'learning'],
            'Travel': ['hotel', 'flight', 'vacation', 'trip', 'travel', 'booking', 'airbnb'],
            'Home & Garden': ['furniture', 'home', 'garden', 'repair', 'maintenance', 'decoration'],
            'Personal Care': ['salon', 'haircut', 'spa', 'cosmetics', 'personal', 'beauty']
        }
        
        # Score each category
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return the category with highest score, or 'Other' if no match
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return 'Other'
    
    def get_expenses_df(self) -> pd.DataFrame:
        """Convert expenses to pandas DataFrame"""
        if not self.expenses:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.expenses)
        df['date'] = pd.to_datetime(df['date'])
        df['amount'] = pd.to_numeric(df['amount'])
        return df
    
    def get_spending_insights(self) -> Dict:
        """Generate expense insights about spending patterns"""
        df = self.get_expenses_df()
        if df.empty:
            return {}
        
        insights = {}
        
        # Total spending
        insights['total_spending'] = df['amount'].sum()
        
        # Average daily spending
        date_range = (df['date'].max() - df['date'].min()).days + 1
        insights['avg_daily_spending'] = insights['total_spending'] / max(1, date_range)
        
        # Top category
        category_spending = df.groupby('category')['amount'].sum()
        insights['top_category'] = category_spending.idxmax()
        insights['top_category_amount'] = category_spending.max()
        insights['top_category_percentage'] = (category_spending.max() / insights['total_spending']) * 100
        
        # Recent trend (last 7 days vs previous 7 days)
        recent_date = df['date'].max()
        last_7_days = df[df['date'] > recent_date - timedelta(days=7)]['amount'].sum()
        prev_7_days = df[(df['date'] <= recent_date - timedelta(days=7)) & 
                        (df['date'] > recent_date - timedelta(days=14))]['amount'].sum()
        
        if prev_7_days > 0:
            insights['trend_percentage'] = ((last_7_days - prev_7_days) / prev_7_days) * 100
        else:
            insights['trend_percentage'] = 0
        
        return insights
   
        
    
# Initialize the expense tracker
if 'tracker' not in st.session_state:
    st.session_state.tracker = ExpenseTracker()

# Category icons mapping
CATEGORY_ICONS = {
    'Food & Dining': '🍽️',
    'Transportation': '🚗',
    'Shopping': '🛍️',
    'Entertainment': '🎬',
    'Health & Medical': '🏥',
    'Utilities': '💡',
    'Education': '📚',
    'Travel': '✈️',
    'Home & Garden': '🏠',
    'Personal Care': '💄',
    'Other': '📝'
}

def get_category_display(category):
    """Get category with icon for display"""
    icon = CATEGORY_ICONS.get(category, '📝')
    return f"{icon} {category}"

# Main app
def main():
    # Enhanced title with styling
    st.markdown('<h1 class="main-title">💰 Expense Tracker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Track your expenses with intelligent categorization and expense insights!</p>', unsafe_allow_html=True)
    
    # Sidebar for adding expenses
    with st.sidebar:
        st.markdown("### 💳 Add New Expense")
        st.markdown("---")

        add_mode = st.radio(
            "How would you like to add expenses?",
            ("Manual entry", "Upload CSV file"),
            help="Choose to add one expense at a time or upload a CSV file."
        )

        if add_mode == "Manual entry":
            with st.form("add_expense_form"):
                st.markdown("#### 💵 Expense Details")

                amount = st.number_input(
                    "💰 Amount (₹)",
                    min_value=0.01,
                    step=0.01,
                    help="Enter the amount you spent"
                )

                description = st.text_input(
                    "📝 Description",
                    placeholder="e.g., Lunch at McDonald's",
                    help="Describe what you bought or paid for"
                )

                date = st.date_input(
                    "📅 Date",
                    value=datetime.now().date(),
                    help="When did this expense occur?"
                )

                # Enhanced category selection with icons
                st.markdown("#### 🏷️ Category")
                categories = ['🤖 Auto-detect'] + [get_category_display(cat) for cat in
                            ['Food & Dining', 'Transportation', 'Shopping', 'Entertainment',
                             'Health & Medical', 'Utilities', 'Education', 'Travel', 'Home & Garden',
                             'Personal Care', 'Other']]

                selected_category = st.selectbox(
                    "Choose or let the system detect",
                    categories,
                    help="Select a category or let the system automatically categorize based on description"
                )

                # Enhanced submit button
                submitted = st.form_submit_button(
                    "✅ Add Expense",
                    help="Click to add this expense to your tracker"
                )

                if submitted and amount and description:
                    # Extract category name without icon
                    if selected_category == '🤖 Auto-detect':
                        category = None
                    else:
                        category = selected_category.split(' ', 1)[1]  # Remove icon

                    expense = st.session_state.tracker.add_expense(
                        amount=amount,
                        description=description,
                        category=category,
                        date=date.strftime("%Y-%m-%d")
                    )

                    # Enhanced success message
                    st.balloons()
                    st.success(f"✅ Added expense: ₹{amount:.2f} - {description}")
                    if category is None:
                        st.info(f"🤖 Categorized as: {get_category_display(expense['category'])}")
                    st.rerun()
        else:
            st.markdown("#### 📁 Upload Expenses CSV File")
            st.info("CSV columns: amount, description, category (optional), date (optional, yyyy-mm-dd)")
            csv_file = st.file_uploader("Upload CSV", type=["csv"])
            if csv_file is not None:
                try:
                    csv_df = pd.read_csv(csv_file)
                    added_count = 0
                    for _, row in csv_df.iterrows():
                        amount = row.get('amount')
                        description = row.get('description')
                        category = row.get('category') if 'category' in row else None
                        date_val = row.get('date') if 'date' in row else None
                        if pd.isna(amount) or pd.isna(description):
                            continue  # skip incomplete rows
                        st.session_state.tracker.add_expense(
                            amount=float(amount),
                            description=str(description),
                            category=str(category) if pd.notna(category) else None,
                            date=str(date_val) if pd.notna(date_val) else None
                        )
                        added_count += 1
                    if added_count:
                        st.success(f"✅ Uploaded and added {added_count} expenses from CSV file!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.warning("No valid rows found in the CSV.")
                except Exception as e:
                    st.error(f"Failed to process CSV: {e}")
        
        # Quick add buttons for common expenses
        st.markdown("---")
        st.markdown("#### ⚡ Quick Add")
        
        quick_expenses = [
            ("☕ Tea", 15.0, "Tea", "Food & Dining"),
            ("⛽ Gas", 30.0, "Gas fill-up", "Transportation"),
            ("🍕 Lunch", 12.0, "Lunch", "Food & Dining"),
            ("🚌 Transit", 3.0, "Public transport", "Transportation")
        ]
        
        cols = st.columns(2)
        for i, (label, amt, desc, cat) in enumerate(quick_expenses):
            with cols[i % 2]:
                if st.button(label, key=f"quick_{i}"):
                    expense = st.session_state.tracker.add_expense(
                        amount=amt,
                        description=desc,
                        category=cat,
                        date=datetime.now().strftime("%Y-%m-%d")
                    )
                    st.success(f"Added {label}!")
                    st.rerun()
    
    # Main content area
    df = st.session_state.tracker.get_expenses_df()
    
    if df.empty:
        st.markdown("""
        <div class="custom-card">
            <h3>🎯 Welcome to Your Expense Tracker!</h3>
            <p>No expenses recorded yet. Get started by:</p>
            <ul>
                <li>📝 Adding your first expense using the sidebar form</li>
                <li>⚡ Using quick-add buttons for common expenses</li>
                <li>🤖 Letting the system automatically categorize your expenses</li>
            </ul>
            <p><strong>Tip:</strong> The more expenses you add, the better insights you'll get!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display enhanced insights
    insights = st.session_state.tracker.get_spending_insights()
    
    # Enhanced metrics with custom styling
    st.markdown("### 📊 Financial Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="custom-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center;">
            <h2 style="margin: 0; color: white;">💰</h2>
            <h3 style="margin: 0.5rem 0; color: white;">₹{:.2f}</h3>
            <p style="margin: 0; opacity: 0.9;">Total Spending</p>
        </div>
        """.format(insights['total_spending']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; text-align: center;">
            <h2 style="margin: 0; color: white;">📅</h2>
            <h3 style="margin: 0.5rem 0; color: white;">₹{:.2f}</h3>
            <p style="margin: 0; opacity: 0.9;">Daily Average</p>
        </div>
        """.format(insights['avg_daily_spending']), unsafe_allow_html=True)
    
    with col3:
        top_category_icon = CATEGORY_ICONS.get(insights['top_category'], '📝')
        st.markdown("""
        <div class="custom-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; text-align: center;">
            <h2 style="margin: 0; color: white;">{}</h2>
            <h3 style="margin: 0.5rem 0; color: white;">{:.1f}%</h3>
            <p style="margin: 0; opacity: 0.9;">{}</p>
        </div>
        """.format(top_category_icon, insights['top_category_percentage'], insights['top_category']), unsafe_allow_html=True)
    
    with col4:
        trend_value = insights['trend_percentage']
        trend_icon = "📈" if trend_value > 0 else "📉" if trend_value < 0 else "📊"
        trend_color = "#ff6b6b" if trend_value > 10 else "#51cf66" if trend_value < -10 else "#339af0"
        st.markdown("""
        <div class="custom-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; text-align: center;">
            <h2 style="margin: 0; color: white;">{}</h2>
            <h3 style="margin: 0.5rem 0; color: white;">{:+.1f}%</h3>
            <p style="margin: 0; opacity: 0.9;">7-Day Trend</p>
        </div>
        """.format(trend_icon, trend_value), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced Charts with containers
    st.markdown("### 📈 Visual Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 🥧 Spending by Category")
        
        category_df = df.groupby('category')['amount'].sum().reset_index()
        # Add icons to category names for the chart
        category_df['category_display'] = category_df['category'].apply(
            lambda x: f"{CATEGORY_ICONS.get(x, '📝')} {x}"
        )
        
        fig_pie = px.pie(
            category_df, 
            values='amount', 
            names='category_display',
            title="",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01),
            margin=dict(t=0, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 📊 Daily Spending Trend")
        
        daily_df = df.groupby('date')['amount'].sum().reset_index()
        fig_line = px.area(
            daily_df, 
            x='date', 
            y='amount',
            title="",
            color_discrete_sequence=['#667eea']
        )
        fig_line.update_traces(fill='tonexty')
        fig_line.update_layout(
            xaxis_title="Date",
            yaxis_title="Amount (₹)",
            margin=dict(t=0, b=40, l=40, r=0),
            showlegend=False
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced Recent expenses table
    st.markdown("### 📈 Recent Expenses")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        num_expenses = st.selectbox(
            "Number of expenses to show:",
            [5, 10, 15, 20],
            index=1
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            ["Date (newest)", "Date (oldest)", "Amount (high)", "Amount (low)"]
        )
    
    recent_df = df.copy()
    
    # Apply sorting
    if sort_by == "Date (newest)":
        recent_df = recent_df.sort_values('date', ascending=False)
    elif sort_by == "Date (oldest)":
        recent_df = recent_df.sort_values('date', ascending=True)
    elif sort_by == "Amount (high)":
        recent_df = recent_df.sort_values('amount', ascending=False)
    else:  # Amount (low)
        recent_df = recent_df.sort_values('amount', ascending=True)
    
    recent_df = recent_df.head(num_expenses)
    
    # Enhanced dataframe with icons and better formatting
    if not recent_df.empty:
        display_df = recent_df[['date', 'description', 'category', 'amount']].copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df['category'] = display_df['category'].apply(get_category_display)
        display_df['amount'] = display_df['amount'].apply(lambda x: f"₹{x:.2f}")
        
        # Rename columns for better display
        display_df.columns = ['📅 Date', '📝 Description', '🏷️ Category', '💰 Amount']
        
        st.dataframe(
            display_df, 
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # Enhanced AI Insights section
    st.markdown("### 🤖 Expense Insights")
    
    if insights:
        # Create tabs for different types of insights
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Trends", "🎯 Recommendations"])
        
        with tab1:
            st.markdown("""
            <div class="insight-card">
                <h4>📊 Spending Summary</h4>
                <p><strong>Total Spending:</strong> ₹{:.2f}</p>
                <p><strong>Top Category:</strong> {} (₹{:.2f} - {:.1f}%)</p>
                <p><strong>Daily Average:</strong> ₹{:.2f}</p>
                <p><strong>Number of Transactions:</strong> {}</p>
            </div>
            """.format(
                insights['total_spending'],
                get_category_display(insights['top_category']),
                insights['top_category_amount'],
                insights['top_category_percentage'],
                insights['avg_daily_spending'],
                len(df)
            ), unsafe_allow_html=True)
        
        with tab2:
            trend_value = insights['trend_percentage']
            
            if trend_value > 10:
                trend_status = "⚠️ Increasing"
                trend_message = f"Your spending has increased by {trend_value:.1f}% in the last 7 days. Consider reviewing your recent purchases."
                trend_color = "#ff6b6b"
            elif trend_value < -10:
                trend_status = "✅ Decreasing"
                trend_message = f"Great job! Your spending has decreased by {abs(trend_value):.1f}% in the last 7 days."
                trend_color = "#51cf66"
            else:
                trend_status = "📊 Stable"
                trend_message = f"Your spending trend is stable with a change of {trend_value:.1f}% in the last 7 days."
                trend_color = "#339af0"
            
            st.markdown("""
            <div class="custom-card" style="border-left-color: {}">
                <h4>📈 7-Day Spending Trend: {}</h4>
                <p>{}</p>
            </div>
            """.format(trend_color, trend_status, trend_message), unsafe_allow_html=True)
            
            # Weekly comparison chart
            if len(df) > 7:
                weekly_df = df.copy()
                weekly_df['week'] = weekly_df['date'].dt.isocalendar().week
                weekly_spending = weekly_df.groupby('week')['amount'].sum().tail(4)
                
                if len(weekly_spending) > 1:
                    fig_weekly = px.bar(
                        x=range(len(weekly_spending)),
                        y=weekly_spending.values,
                        title="Weekly Spending Comparison (Last 4 Weeks)",
                        labels={'x': 'Week', 'y': 'Amount (₹)'},
                        color_discrete_sequence=['#667eea']
                    )
                    fig_weekly.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
                    st.plotly_chart(fig_weekly, use_container_width=True)
        
        with tab3:
            # Generate personalized recommendations
            category_spending = df.groupby('category')['amount'].sum().sort_values(ascending=False)
            top_categories = category_spending.head(3)
            
            recommendations = []
            
            # Budget recommendations
            if insights['avg_daily_spending'] > 50:
                recommendations.append("🎯 Consider setting a daily spending limit of ₹40-45 to reduce expenses.")
            
            # Category-specific recommendations
            for category, amount in top_categories.items():
                percentage = (amount / insights['total_spending']) * 100
                if percentage > 40:
                    if category == 'Food & Dining':
                        recommendations.append(f"🍽️ {category} represents {percentage:.1f}% of your spending. Try meal planning or cooking at home more often.")
                    elif category == 'Shopping':
                        recommendations.append(f"🛍️ {category} is {percentage:.1f}% of your spending. Consider implementing a 24-hour rule before making purchases.")
                    elif category == 'Entertainment':
                        recommendations.append(f"🎬 {category} accounts for {percentage:.1f}% of spending. Look for free or low-cost entertainment alternatives.")
                    else:
                        recommendations.append(f"📊 {category} is your highest expense at {percentage:.1f}%. Review if all expenses in this category are necessary.")
            
            # Trend-based recommendations
            if trend_value > 20:
                recommendations.append("⚠️ Your spending has increased significantly. Review your recent purchases and identify areas to cut back.")
            elif trend_value < -20:
                recommendations.append("✅ Excellent spending discipline! Consider putting the saved money into savings or investments.")
            
            if not recommendations:
                recommendations.append("🎆 Your spending looks well-balanced! Keep up the good financial habits.")
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")
    
    st.markdown("---")
    
    # Enhanced export section
    st.markdown("### 📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as CSV", help="Download all expense data as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"expenses_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Generate Report", help="Create a detailed spending report"):
            # Generate a comprehensive report
            report = f"""
# Expense Report - {datetime.now().strftime('%B %Y')}

## Summary
- Total Expenses: ₹{insights['total_spending']:.2f}
- Number of Transactions: {len(df)}
- Average Daily Spending: ₹{insights['avg_daily_spending']:.2f}
- Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}

## Top Categories
{category_spending.head().to_string()}

## Recent Transactions
{df.tail(10)[['date', 'description', 'category', 'amount']].to_string(index=False)}
            """
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"expense_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    with col3:
        confirm_clear = st.checkbox("I understand this will delete all my data", key="clear_confirm")
        clear_btn = st.button("🔄 Clear All Data", help="Remove all expense data (cannot be undone)", disabled=not confirm_clear)
        if clear_btn and confirm_clear:
            st.session_state.tracker.expenses = []
            st.session_state.tracker.save_data()
            st.success("All data cleared!")
            st.session_state.clear_confirm = False
            st.rerun()

if __name__ == "__main__":
    main()
