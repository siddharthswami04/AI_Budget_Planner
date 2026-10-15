"""
AI Personal Budget Planner - single-file Streamlit app
======================================================

Everything lives in this one file: UI, data management, analytics,
forecasting, savings goals, a LangChain agent, Gemini integration and AI
transaction categorisation. Sample data is embedded below (SAMPLE_DATA_CSV).

Quick start
-----------
    pip install -r requirements.txt
    streamlit run app.py

Opens at http://localhost:8501

Data & AI
---------
- On first launch (no transactions.csv) the embedded sample data is loaded.
  Changes are saved locally to transactions.csv and goals.csv.
- Paste your Gemini API key in the sidebar and click "Connect AI". The key is
  kept only in memory for the current session - nothing is stored on disk and
  no environment variables or .env files are used. Core budgeting works
  without AI.
- Architecture: Streamlit UI -> LangChain agent -> deterministic Pandas tools
  -> Gemini explanation. All money maths is done in Python, not by the LLM.
- Models offered: gemini-3.5-flash and gemini-3.5-flash-lite (see DEFAULT_MODELS).
"""

import os
import io
import json
import math
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

# Optional AI imports. The app still works without them for non-AI features.
try:
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_google_genai import ChatGoogleGenerativeAI
    AI_PACKAGES_OK = True
except Exception:
    AI_PACKAGES_OK = False


# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "AI Personal Budget Planner"
DATA_FILE = "transactions.csv"
GOAL_FILE = "goals.csv"

CATEGORIES = [
    "Food", "Transport", "Rent", "Bills", "Education", "Shopping",
    "Entertainment", "Healthcare", "Travel", "Salary", "Freelance",
    "Other"
]

PAYMENT_METHODS = ["Cash", "UPI", "Debit Card", "Credit Card", "Bank Transfer", "Other"]

# Only two Gemini models are offered:
# - gemini-3.5-flash:      smarter, better for detailed financial questions
# - gemini-3.5-flash-lite: fastest and cheapest, good for quick answers
DEFAULT_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

SAMPLE_DATA_CSV = """id,date,description,category,amount,type,payment_method
1,2026-07-01,Salary,Salary,55000,Income,Bank Transfer
2,2026-07-02,Monthly Rent,Rent,12000,Expense,Bank Transfer
3,2026-07-03,Swiggy dinner,Food,450,Expense,UPI
4,2026-07-04,Uber to college,Transport,220,Expense,UPI
5,2026-07-05,College fees,Education,5000,Expense,Bank Transfer
6,2026-07-07,Electricity bill,Bills,1800,Expense,UPI
7,2026-07-10,Amazon shopping,Shopping,2400,Expense,Debit Card
8,2026-07-12,Netflix,Entertainment,649,Expense,Debit Card
9,2026-07-15,Freelance project,Freelance,8000,Income,Bank Transfer
10,2026-07-18,Restaurant dinner,Food,1200,Expense,UPI
11,2026-08-01,Salary,Salary,55000,Income,Bank Transfer
12,2026-08-02,Monthly Rent,Rent,12000,Expense,Bank Transfer
13,2026-08-04,Zomato lunch,Food,520,Expense,UPI
14,2026-08-06,Metro recharge,Transport,600,Expense,UPI
15,2026-08-08,Online course,Education,1800,Expense,Debit Card

"""

TX_COLUMNS = ["id", "date", "description", "category", "amount", "type", "payment_method"]

GOAL_COLUMNS = ["name", "target", "saved"]


# ============================================================
# DATA HELPERS
# ============================================================

def empty_transactions():
    return pd.DataFrame(columns=TX_COLUMNS)


def normalize_transactions(df):
    if df is None or df.empty:
        return empty_transactions()

    df = df.copy()

    # Handle common alternative column names.
    aliases = {
        "Date": "date",
        "Description": "description",
        "Category": "category",
        "Amount": "amount",
        "Type": "type",
        "Payment Method": "payment_method",
        "payment": "payment_method",
        "transaction_type": "type",
    }
    df = df.rename(columns=aliases)

    for col in TX_COLUMNS:
        if col not in df.columns:
            if col == "id":
                df[col] = range(1, len(df) + 1)
            elif col == "type":
                df[col] = "Expense"
            elif col == "payment_method":
                df[col] = "Other"
            else:
                df[col] = ""

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["description"] = df["description"].fillna("").astype(str)
    df["category"] = df["category"].fillna("Other").astype(str)
    df["type"] = df["type"].fillna("Expense").astype(str).str.title()
    df["payment_method"] = df["payment_method"].fillna("Other").astype(str)

    df.loc[~df["type"].isin(["Income", "Expense"]), "type"] = "Expense"

    # Repair invalid/missing dates.
    if df["date"].isna().any():
        df.loc[df["date"].isna(), "date"] = pd.Timestamp.today().normalize()

    # IDs need to be unique.
    ids = pd.to_numeric(df["id"], errors="coerce")
    df["id"] = ids
    if df["id"].isna().any() or df["id"].duplicated().any():
        df["id"] = range(1, len(df) + 1)
    df["id"] = df["id"].astype(int)

    return df[TX_COLUMNS].sort_values("date", ascending=False).reset_index(drop=True)


def load_sample_transactions():
    return normalize_transactions(pd.read_csv(io.StringIO(SAMPLE_DATA_CSV.strip())))


def load_transactions():
    if os.path.exists(DATA_FILE):
        try:
            return normalize_transactions(pd.read_csv(DATA_FILE))
        except Exception:
            return empty_transactions()

    # First launch (no saved file yet): use the embedded sample data.
    return load_sample_transactions()


def save_transactions(df):
    normalize_transactions(df).to_csv(DATA_FILE, index=False)


def next_id(df):
    if df.empty:
        return 1
    return int(pd.to_numeric(df["id"], errors="coerce").max()) + 1


def load_goals():
    if not os.path.exists(GOAL_FILE):
        return pd.DataFrame(columns=GOAL_COLUMNS)
    try:
        df = pd.read_csv(GOAL_FILE)
        for c in GOAL_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        df["name"] = df["name"].astype(str)
        df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0.0)
        df["saved"] = pd.to_numeric(df["saved"], errors="coerce").fillna(0.0)
        return df[GOAL_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=GOAL_COLUMNS)


def save_goals(df):
    df[GOAL_COLUMNS].to_csv(GOAL_FILE, index=False)


# ============================================================
# FINANCIAL ANALYSIS
# ============================================================

def income_total(df):
    return float(df.loc[df["type"] == "Income", "amount"].sum())


def expense_total(df):
    return float(df.loc[df["type"] == "Expense", "amount"].sum())


def net_balance(df):
    return income_total(df) - expense_total(df)


def category_spending(df):
    exp = df[df["type"] == "Expense"]
    if exp.empty:
        return pd.DataFrame(columns=["category", "amount"])
    return (
        exp.groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )


def monthly_summary(df):
    if df.empty:
        return pd.DataFrame(columns=["month", "income", "expenses", "savings"])
    x = df.copy()
    x["month"] = x["date"].dt.to_period("M").astype(str)
    inc = x[x["type"] == "Income"].groupby("month")["amount"].sum()
    exp = x[x["type"] == "Expense"].groupby("month")["amount"].sum()
    out = pd.DataFrame({"income": inc, "expenses": exp}).fillna(0)
    out["savings"] = out["income"] - out["expenses"]
    return out.reset_index()


def top_expenses(df, n=5):
    exp = df[df["type"] == "Expense"].copy()
    return exp.sort_values("amount", ascending=False).head(n)


def unusual_expenses(df):
    exp = df[df["type"] == "Expense"].copy()
    if len(exp) < 3:
        return exp.iloc[0:0].copy()
    mean = exp["amount"].mean()
    std = exp["amount"].std()
    threshold = mean + (2 * std if pd.notna(std) else 0)
    return exp[exp["amount"] > threshold].sort_values("amount", ascending=False)


def forecast_expenses(df):
    monthly = monthly_summary(df)
    if len(monthly) < 2:
        return None, "Forecast needs at least 2 months of transaction data."
    exp = monthly["expenses"].tail(3)
    if len(df[df["type"] == "Expense"]) < 5:
        return None, "Forecast needs at least 5 expense transactions."
    prediction = float(exp.mean())
    return prediction, "Forecast based on the average of the latest available months."


def savings_goal_status(goals):
    if goals.empty:
        return goals.copy()
    x = goals.copy()
    x["remaining"] = (x["target"] - x["saved"]).clip(lower=0)
    x["progress"] = x.apply(
        lambda r: min(100.0, r["saved"] / r["target"] * 100) if r["target"] > 0 else 0,
        axis=1
    )
    return x


# ============================================================
# AI CATEGORY CLASSIFIER
# ============================================================

CATEGORY_RULES = {
    "food": ["swiggy", "zomato", "restaurant", "cafe", "food", "pizza", "burger", "grocery"],
    "transport": ["uber", "ola", "metro", "bus", "taxi", "fuel", "petrol", "diesel", "rapido"],
    "rent": ["rent", "hostel", "pg"],
    "bills": ["electricity", "water bill", "internet", "wifi", "mobile", "recharge", "bill"],
    "education": ["college", "course", "book", "udemy", "coursera", "education", "exam"],
    "shopping": ["amazon", "flipkart", "myntra", "shopping", "clothes", "shoes"],
    "entertainment": ["movie", "netflix", "spotify", "game", "gaming", "entertainment"],
    "healthcare": ["hospital", "doctor", "medicine", "pharmacy", "health"],
    "travel": ["flight", "hotel", "train", "travel", "trip", "airbnb"],
    "salary": ["salary", "paycheck"],
    "freelance": ["freelance", "client", "project payment"],
}


def rule_based_category(text):
    t = str(text).lower()
    for category, words in CATEGORY_RULES.items():
        if any(w in t for w in words):
            return category.title()
    return "Other"


def ai_categorize(description, api_key, model_name):
    if not api_key or not AI_PACKAGES_OK:
        return rule_based_category(description)

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
        )
        prompt = (
            "Classify this transaction into exactly one category from this list: "
            + ", ".join(CATEGORIES)
            + ". Return only the category name. Transaction: "
            + str(description)
        )
        response = llm.invoke(prompt)
        text = getattr(response, "content", str(response)).strip()
        text = text.replace("*", "").strip()
        for c in CATEGORIES:
            if text.lower() == c.lower():
                return c
        return rule_based_category(description)
    except Exception:
        return rule_based_category(description)


# ============================================================
# LANGCHAIN AGENT
# ============================================================

def build_agent(df, goals, api_key, model_name):
    if not AI_PACKAGES_OK:
        return None, "LangChain AI packages are not installed correctly."

    if not api_key:
        return None, "Gemini API key is not connected."

    # Snapshot data so tools use the currently displayed dataset.
    data = normalize_transactions(df.copy())
    goal_data = goals.copy()

    @tool
    def get_financial_summary() -> str:
        """Get total income, expenses and net balance from the user's transactions."""
        return json.dumps({
            "total_income": round(income_total(data), 2),
            "total_expenses": round(expense_total(data), 2),
            "net_balance": round(net_balance(data), 2),
            "transaction_count": int(len(data))
        })

    @tool
    def get_category_spending() -> str:
        """Get spending totals grouped by expense category."""
        return category_spending(data).to_json(orient="records")

    @tool
    def get_monthly_comparison() -> str:
        """Get monthly income, expenses and savings for comparison."""
        return monthly_summary(data).to_json(orient="records")

    @tool
    def get_top_expenses() -> str:
        """Get the five largest expense transactions."""
        x = top_expenses(data, 5).copy()
        if x.empty:
            return "No expenses found."
        x["date"] = x["date"].dt.strftime("%Y-%m-%d")
        return x[["date", "description", "category", "amount"]].to_json(orient="records")

    @tool
    def get_unusual_expenses() -> str:
        """Detect unusually large expenses using a simple statistical threshold."""
        x = unusual_expenses(data).copy()
        if x.empty:
            return "No statistically unusual expenses found."
        x["date"] = x["date"].dt.strftime("%Y-%m-%d")
        return x[["date", "description", "category", "amount"]].to_json(orient="records")

    @tool
    def get_expense_forecast() -> str:
        """Forecast next monthly expenses from recent transaction history."""
        prediction, message = forecast_expenses(data)
        if prediction is None:
            return message
        return json.dumps({
            "forecast_monthly_expenses": round(prediction, 2),
            "method": message
        })

    @tool
    def get_savings_goals() -> str:
        """Get savings goals, progress and remaining target amounts."""
        x = savings_goal_status(goal_data)
        if x.empty:
            return "No savings goals created."
        return x.to_json(orient="records")

    tools = [
        get_financial_summary,
        get_category_spending,
        get_monthly_comparison,
        get_top_expenses,
        get_unusual_expenses,
        get_expense_forecast,
        get_savings_goals,
    ]

    system_prompt = """
You are the AI assistant inside a personal budget planner.
Use the provided tools whenever the user asks for numerical facts about their finances.
Never invent transaction values or financial statistics.
Explain results clearly and concisely.
Currency is Indian Rupees (₹).
You may suggest budgeting and saving habits, but do not present yourself as a professional
investment, tax, loan, or regulated financial adviser.
If there is insufficient data, say so.
When comparing months/categories, use the tool results.
"""

    try:
        agent = create_agent(
            model=ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.2,
            ),
            tools=tools,
            system_prompt=system_prompt,
        )
        return agent, None
    except Exception as e:
        return None, str(e)


def ask_agent(agent, question):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", str(last))
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)
    return "I couldn't generate a response."


# ============================================================
# STREAMLIT SETUP
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💰",
    layout="wide",
)

if "transactions" not in st.session_state:
    st.session_state.transactions = load_transactions()

if "goals" not in st.session_state:
    st.session_state.goals = load_goals()

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "model" not in st.session_state:
    st.session_state.model = DEFAULT_MODELS[0]

if "ai_connected" not in st.session_state:
    st.session_state.ai_connected = bool(st.session_state.api_key)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


df = normalize_transactions(st.session_state.transactions)
goals = st.session_state.goals


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💰 Budget Planner")
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Transactions",
        "Financial Goals",
        "AI Assistant",
        "Forecast",
        "Settings",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("🤖 AI Configuration")

api_input = st.sidebar.text_input(
    "Gemini API Key",
    value=st.session_state.api_key,
    type="password",
    help="Used only during this session. It is never saved to disk.",
)

model_input = st.sidebar.selectbox(
    "Gemini Model",
    DEFAULT_MODELS,
    index=DEFAULT_MODELS.index(st.session_state.model)
    if st.session_state.model in DEFAULT_MODELS else 0,
)

if st.sidebar.button("Connect AI", use_container_width=True):
    st.session_state.api_key = api_input.strip()
    st.session_state.model = model_input
    st.session_state.ai_connected = bool(st.session_state.api_key)
    st.rerun()

if st.session_state.ai_connected:
    st.sidebar.success("AI key connected for this session.")
else:
    st.sidebar.info("AI features require a Gemini key. Core budgeting works without AI.")

if st.sidebar.button("Refresh Data", use_container_width=True):
    st.session_state.transactions = load_transactions()
    st.session_state.goals = load_goals()
    st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":
    st.title("💰 AI Personal Budget Planner")
    st.caption("Track, analyze and understand your personal finances.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Income", f"₹{income_total(df):,.2f}")
    c2.metric("Total Expenses", f"₹{expense_total(df):,.2f}")
    c3.metric("Balance", f"₹{net_balance(df):,.2f}")
    savings_rate = (net_balance(df) / income_total(df) * 100) if income_total(df) else 0
    c4.metric("Savings Rate", f"{savings_rate:.1f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Spending by Category")
        cat = category_spending(df)
        if cat.empty:
            st.info("No expense data yet.")
        else:
            fig = px.pie(cat, names="category", values="amount", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Monthly Income vs Expenses")
        monthly = monthly_summary(df)
        if monthly.empty:
            st.info("Add transactions to see the chart.")
        else:
            plot_df = monthly.melt(
                id_vars="month",
                value_vars=["income", "expenses"],
                var_name="type",
                value_name="amount",
            )
            fig = px.bar(plot_df, x="month", y="amount", color="type", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Transactions")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# ============================================================
# TRANSACTIONS
# ============================================================

elif page == "Transactions":
    st.title("📋 Transactions")

    tab1, tab2, tab3 = st.tabs(["Add Transaction", "Manage Transactions", "Import / Export"])

    with tab1:
        st.subheader("Add a transaction")
        with st.form("add_transaction_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            tx_date = col1.date_input("Date", value=date.today())
            tx_type = col2.selectbox("Type", ["Expense", "Income"])
            description = st.text_input("Description", placeholder="e.g. Swiggy dinner")
            amount = st.number_input("Amount (₹)", min_value=0.0, step=100.0)
            col3, col4 = st.columns(2)

            if tx_type == "Income":
                default_cat = "Salary"
            else:
                default_cat = rule_based_category(description) if description else "Other"

            category = col3.selectbox(
                "Category",
                CATEGORIES,
                index=CATEGORIES.index(default_cat) if default_cat in CATEGORIES else 0,
            )
            payment = col4.selectbox("Payment Method", PAYMENT_METHODS)

            ai_cat = st.checkbox(
                "Use AI/rule-based category suggestion",
                value=False,
                disabled=not bool(st.session_state.api_key),
            )

            submitted = st.form_submit_button("Add Transaction", type="primary")

        if submitted:
            if amount <= 0:
                st.error("Amount must be greater than 0.")
            elif not description.strip():
                st.error("Please enter a description.")
            else:
                final_category = category
                if ai_cat:
                    final_category = ai_categorize(
                        description,
                        st.session_state.api_key,
                        st.session_state.model,
                    )

                new_row = {
                    "id": next_id(df),
                    "date": pd.Timestamp(tx_date),
                    "description": description.strip(),
                    "category": final_category,
                    "amount": float(amount),
                    "type": tx_type,
                    "payment_method": payment,
                }
                st.session_state.transactions = pd.concat(
                    [df, pd.DataFrame([new_row])],
                    ignore_index=True,
                )
                save_transactions(st.session_state.transactions)
                st.success(f"Transaction added as {final_category}.")
                st.rerun()

    with tab2:
        if df.empty:
            st.info("No transactions available.")
        else:
            search = st.text_input("Search", placeholder="Search description/category")
            type_filter = st.multiselect("Type", ["Income", "Expense"], default=["Income", "Expense"])
            cat_filter = st.multiselect("Category", CATEGORIES)

            filtered = df.copy()
            if search:
                mask = (
                    filtered["description"].str.contains(search, case=False, na=False)
                    | filtered["category"].str.contains(search, case=False, na=False)
                )
                filtered = filtered[mask]
            if type_filter:
                filtered = filtered[filtered["type"].isin(type_filter)]
            if cat_filter:
                filtered = filtered[filtered["category"].isin(cat_filter)]

            st.dataframe(filtered, use_container_width=True, hide_index=True)

            st.subheader("Delete a transaction")
            delete_id = st.number_input(
                "Transaction ID",
                min_value=1,
                step=1,
                value=int(filtered.iloc[0]["id"]) if not filtered.empty else 1,
            )
            if st.button("Delete Selected Transaction", type="secondary"):
                if delete_id in df["id"].values:
                    st.session_state.transactions = df[df["id"] != delete_id].copy()
                    save_transactions(st.session_state.transactions)
                    st.success("Transaction deleted.")
                    st.rerun()
                else:
                    st.error("Transaction ID not found.")

    with tab3:
        st.subheader("Import CSV")
        uploaded = st.file_uploader("Upload a transaction CSV", type=["csv"])
        if uploaded is not None:
            try:
                imported = normalize_transactions(pd.read_csv(uploaded))
                st.write(f"Found {len(imported)} rows.")
                if st.button("Replace Current Data With Imported CSV"):
                    st.session_state.transactions = imported
                    save_transactions(imported)
                    st.success("CSV imported successfully.")
                    st.rerun()
            except Exception as e:
                st.error(f"Could not import CSV: {e}")

        st.subheader("Export")
        csv_bytes = normalize_transactions(df).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Transactions CSV",
            data=csv_bytes,
            file_name="transactions.csv",
            mime="text/csv",
        )


# ============================================================
# FINANCIAL GOALS
# ============================================================

elif page == "Financial Goals":
    st.title("🏆 Financial Goals")

    with st.form("goal_form", clear_on_submit=True):
        name = st.text_input("Goal Name", placeholder="e.g. New Laptop")
        target = st.number_input("Target Amount (₹)", min_value=0.0, step=1000.0)
        saved = st.number_input("Already Saved (₹)", min_value=0.0, step=500.0)
        submitted = st.form_submit_button("Create Goal", type="primary")

    if submitted:
        if not name.strip() or target <= 0:
            st.error("Enter a goal name and a target greater than 0.")
        else:
            new = pd.concat(
                [goals, pd.DataFrame([{
                    "name": name.strip(),
                    "target": target,
                    "saved": min(saved, target),
                }])],
                ignore_index=True,
            )
            st.session_state.goals = new
            save_goals(new)
            st.success("Goal created.")
            st.rerun()

    status = savings_goal_status(goals)

    if status.empty:
        st.info("No financial goals yet.")
    else:
        for i, row in status.iterrows():
            st.subheader(row["name"])
            st.write(
                f"Saved: ₹{row['saved']:,.0f} / ₹{row['target']:,.0f} "
                f"— {row['progress']:.1f}%"
            )
            st.progress(min(1.0, row["progress"] / 100))

            remaining = row["remaining"]
            if remaining <= 0:
                st.success("Goal completed!")
            else:
                st.caption(f"Remaining: ₹{remaining:,.2f}")

    st.subheader("Update / Delete")
    if not goals.empty:
        selected_goal = st.selectbox("Goal", goals["name"].tolist())
        idx = goals.index[goals["name"] == selected_goal][0]

        new_saved = st.number_input(
            "Update saved amount (₹)",
            min_value=0.0,
            value=float(goals.loc[idx, "saved"]),
            step=500.0,
        )

        c1, c2 = st.columns(2)
        if c1.button("Update Saved Amount"):
            new = goals.copy()
            new.loc[idx, "saved"] = min(new_saved, float(new.loc[idx, "target"]))
            st.session_state.goals = new
            save_goals(new)
            st.success("Goal updated.")
            st.rerun()

        if c2.button("Delete Goal"):
            new = goals.drop(index=idx).reset_index(drop=True)
            st.session_state.goals = new
            save_goals(new)
            st.success("Goal deleted.")
            st.rerun()


# ============================================================
# AI ASSISTANT
# ============================================================

elif page == "AI Assistant":
    st.title("💬 AI Financial Assistant")
    st.caption("Ask questions about the transactions and savings goals stored in this app.")

    if not st.session_state.api_key:
        st.warning("Connect your Gemini API key from the sidebar first.")
    elif not AI_PACKAGES_OK:
        st.error("AI packages are unavailable. Install the dependencies listed at the top of app.py.")
    else:
        agent, err = build_agent(
            df,
            goals,
            st.session_state.api_key,
            st.session_state.model,
        )

        if err:
            st.error(err)
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            question = st.chat_input(
                "Ask: How much did I spend on food?"
            )

            if question:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question,
                })
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing your financial data..."):
                        try:
                            answer = ask_agent(agent, question)
                            st.markdown(answer)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                            })
                        except Exception as e:
                            error_text = f"AI request failed: {e}"
                            st.error(error_text)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": error_text,
                            })

            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()


# ============================================================
# FORECAST
# ============================================================

elif page == "Forecast":
    st.title("📈 Expense Forecast")

    prediction, message = forecast_expenses(df)

    if prediction is None:
        st.warning(message)
    else:
        st.metric("Estimated Monthly Expenses", f"₹{prediction:,.2f}")
        st.caption(message)

        monthly = monthly_summary(df)
        if not monthly.empty:
            fig = px.line(
                monthly,
                x="month",
                y="expenses",
                markers=True,
                title="Historical Monthly Expenses",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                monthly.style.format({
                    "income": "₹{:,.2f}",
                    "expenses": "₹{:,.2f}",
                    "savings": "₹{:,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.info(
            "This is a simple budgeting forecast based on recent expense history. "
            "It is not a guarantee of future spending."
        )


# ============================================================
# SETTINGS
# ============================================================

elif page == "Settings":
    st.title("⚙️ Settings & Privacy")

    st.subheader("Data")
    st.write(f"Transactions: **{len(df)}**")
    st.write(f"Financial goals: **{len(goals)}**")

    st.subheader("AI")
    st.write(f"AI package status: **{'Available' if AI_PACKAGES_OK else 'Not available'}**")
    st.write(f"Connected this session: **{'Yes' if st.session_state.api_key else 'No'}**")
    st.write(f"Selected model: **{st.session_state.model}**")

    st.subheader("Privacy")
    st.write(
        "Your transactions are processed locally by this Streamlit app. "
        "When you use Gemini AI, relevant financial data is sent to Google's API "
        "to generate the requested AI response. Your API key is entered as a "
        "password field and is not written to the transaction CSV."
    )

    st.subheader("Danger Zone")
    if st.button("Delete All Local Transactions"):
        st.session_state.transactions = empty_transactions()
        save_transactions(st.session_state.transactions)
        st.success("All local transactions deleted.")
        st.rerun()

    if st.button("Load Sample Data (replaces current transactions)"):
        st.session_state.transactions = load_sample_transactions()
        save_transactions(st.session_state.transactions)
        st.success("Sample data loaded.")
        st.rerun()

    if st.button("Reset Goals"):
        st.session_state.goals = pd.DataFrame(columns=GOAL_COLUMNS)
        save_goals(st.session_state.goals)
        st.success("Goals reset.")
        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()
st.sidebar.caption("AI Personal Budget Planner • Python + Streamlit + Pandas + Plotly + LangChain + Gemini")
