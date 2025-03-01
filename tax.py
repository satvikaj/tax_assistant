import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
import re
import time
import json
from datetime import datetime

# Configure Gemini
genai.configure(api_key="AIzaSyALSp9jVTV04ziisP5IdFu5-VpPsx39NfU")
model = genai.GenerativeModel('gemini-pro')

class FinanceAgent:
    def __init__(self):
        self.financial_terms = {
            'income_statement': ['revenue', 'gross profit', 'operating income', 'net income'],
            'balance_sheet': ['assets', 'liabilities', 'equity', 'debt'],
            'cash_flow': ['operating cash flow', 'investing cash flow', 'financing cash flow']
        }
        self.required_metrics = ['revenue', 'net income', 'assets', 'liabilities', 'equity']
       
    def analyze_with_gemini(self, text, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt + text)
                return response.text
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return f"API Error: {str(e)}"
        return "System busy - please try later"

    def process_financial_doc(self, uploaded_file):
        if uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            text = '\n'.join([page.extract_text() for page in reader.pages])
            return self._analyze_financial_text(text)
        return {"error": "Unsupported file format"}

    def _analyze_financial_text(self, text):
        findings = {}
        for category, terms in self.financial_terms.items():
            pattern = r'(?i)({}):?\s*([\$€£]?[\d,\-+.]+[\d])\b'.format('|'.join(terms))
            matches = re.findall(pattern, text)
            if matches:
                findings[category] = {}
                for match in matches:
                    key = match[0].lower().strip()
                    value = self._clean_currency(match[1])
                    findings[category][key] = value
       
        # Validate required metrics
        for metric in self.required_metrics:
            found = False
            for category in findings.values():
                if metric in category:
                    found = True
                    break
            if not found:
                findings.setdefault('income_statement', {})[metric] = 'N/A'
       
        analysis_prompt = """Analyze this financial document:
        1. Identify 3 key trends with percentage changes
        2. Highlight top 2 risks with financial impact estimates
        3. Calculate and interpret these ratios: ROE, ROI, Debt/Equity
        4. Provide 3 actionable recommendations
        Format response with clear headings and bullet points"""
       
        return {
            "structured_data": findings,
            "ai_analysis": self.analyze_with_gemini(analysis_prompt, text)
        }

    def _clean_currency(self, value):
        try:
            cleaned = re.sub(r'[^\d\.\-+]', '', value)
           
            if not cleaned or cleaned in ('.', '-', '+'):
                return 0.0
               
            if cleaned.count('.') > 1:
                cleaned = cleaned.replace('.', '', cleaned.count('.')-1)
               
            if cleaned.startswith('.') or cleaned.endswith('.'):
                cleaned = cleaned.strip('.')
               
            return float(cleaned) if cleaned else 0.0
        except Exception as e:
            print(f"Currency conversion error: {e}")
            return 0.0

    def stock_analyzer(self, ticker):
        prompt = f"""Analyze {ticker} stock with:
        - Current valuation metrics
        - Technical indicators (RSI, MACD)
        - Fundamental analysis (P/E ratio, EPS growth)
        - 12-month price target
        Output as JSON with keys: valuation, technical, fundamentals, price_target"""
        try:
            response = self.analyze_with_gemini("", prompt)
            return json.loads(response.replace("```json", "").replace("```", ""))
        except Exception as e:
            return {"error": f"Stock analysis failed: {str(e)}"}

    def expense_tracker(self, receipt_text):
        prompt = f"""Extract structured data from receipt:
        {{
            "vendor": "",
            "date": "YYYY-MM-DD",
            "amount": 0.00,
            "category": "",
            "tax_details": {{
                "gst": 0.00,
                "total_tax": 0.00
            }}
        }}
        From text: {receipt_text}"""
        try:
            response = self.analyze_with_gemini("", prompt)
            return json.loads(response.replace("```json", "").replace("```", ""))
        except Exception as e:
            return {"error": f"Receipt processing failed: {str(e)}"}

# Streamlit Interface
st.set_page_config(page_title="AI Finance Agent", layout="wide")
st.title("💰 Financial Intelligence Assistant")

tab1, tab2, tab3, tab4 = st.tabs(["Document Analysis", "Stock Insights", "Expense Tracking", "Fraud Detection"])

with tab1:
    st.subheader("Financial Document Analysis")
    uploaded_file = st.file_uploader("Upload PDF (Annual Report, Balance Sheet)", type=['pdf'])
   
    if uploaded_file:
        agent = FinanceAgent()
        with st.spinner('Analyzing document...'):
            try:
                report = agent.process_financial_doc(uploaded_file)
               
                if "error" in report:
                    st.error(report["error"])
                else:
                    col1, col2 = st.columns([1, 2])
                   
                    with col1:
                        st.subheader("Financial Metrics")
                        for category, data in report["structured_data"].items():
                            with st.expander(category.replace('_', ' ').title()):
                                st.json(data)
                   
                    with col2:
                        st.subheader("Expert Analysis")
                        st.markdown(report["ai_analysis"])
                       
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

with tab2:
    st.subheader("Stock Market Analysis")
    ticker = st.text_input("Enter stock ticker (e.g., AAPL):", key="ticker_input")
   
    if st.button("Analyze Stock"):
        agent = FinanceAgent()
        with st.spinner('Generating insights...'):
            analysis = agent.stock_analyzer(ticker)
           
            if "error" in analysis:
                st.error(analysis["error"])
            else:
                st.subheader(f"{ticker.upper()} Analysis")
               
                cols = st.columns(3)
                cols[0].metric("Current Price", analysis.get("valuation", {}).get("current_price", "N/A"))
                cols[1].metric("P/E Ratio", analysis.get("fundamentals", {}).get("pe_ratio", "N/A"))
                cols[2].metric("Price Target", analysis.get("price_target", "N/A"))
               
                with st.expander("Technical Analysis"):
                    st.write(analysis.get("technical", {}))
               
                with st.expander("Fundamental Analysis"):
                    st.write(analysis.get("fundamentals", {}))

with tab3:
    st.subheader("Expense Management")
    receipt_text = st.text_area("Paste receipt text or upload PDF:", key="receipt_input")
   
    if st.button("Process Expense"):
        agent = FinanceAgent()
        with st.spinner('Categorizing expense...'):
            result = agent.expense_tracker(receipt_text)
           
            if "error" in result:
                st.error(result["error"])
            else:
                st.subheader("Expense Breakdown")
                cols = st.columns(4)
                cols[0].metric("Vendor", result.get("vendor", "N/A"))
                cols[1].metric("Amount", f"${result.get('amount', 0):.2f}")
                cols[2].metric("Category", result.get("category", "N/A"))
                cols[3].metric("Date", result.get("date", "N/A"))
               
                st.markdown("**Tax Details**")
                tax_cols = st.columns(2)
                tax_cols[0].metric("GST", f"${result.get('tax_details', {}).get('gst', 0):.2f}")
                tax_cols[1].metric("Total Tax", f"${result.get('tax_details', {}).get('total_tax', 0):.2f}")

with tab4:
    st.subheader("Transaction Monitoring")
    transaction_data = st.text_area("Enter transaction details:", key="transaction_input")
   
    if st.button("Check Fraud Risk"):
        agent = FinanceAgent()
        with st.spinner('Analyzing patterns...'):
            prompt = f"""Analyze transaction for fraud risks:
            {transaction_data}
            Output JSON with:
            {{
                "risk_score": 0-100,
                "risk_level": "low/medium/high",
                "suspicious_patterns": [],
                "recommended_actions": []
            }}"""
            try:
                response = agent.analyze_with_gemini(prompt, "")
                result = json.loads(response.replace("```json", "").replace("```", ""))
               
                st.subheader("Risk Assessment")
                risk_score = result.get("risk_score", 0)
                risk_level = result.get("risk_level", "low")
                risk_color = {"low": "green", "medium": "orange", "high": "red"}.get(risk_level, "gray")
               
                cols = st.columns(2)
                cols[0].metric("Risk Score", f"{risk_score}/100")
                cols[1].metric("Risk Level", f"{risk_level.title()}", delta_color="off")
                cols[1].markdown(f"<span style='color:{risk_color};font-size:24px'>●</span>",
                              unsafe_allow_html=True)
               
                with st.expander("Details"):
                    st.write("Suspicious Patterns:", result.get("suspicious_patterns", []))
                    st.write("Recommended Actions:", result.get("recommended_actions", []))
                   
            except Exception as e:
                st.error(f"Fraud analysis failed: {str(e)}")

st.divider()
st.caption("⚠️ Disclaimer: This tool provides informational insights only, not financial advice. Always consult a qualified professional.")
