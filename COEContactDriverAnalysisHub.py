import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import io

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Load CSS for custom styling
file_path = 'style.css'
if os.path.exists(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
else:
    st.error('style.css file not found. Please check the file path.')
    st.stop()

st.sidebar.header('📞 `COE` Contact Driver Analysis Hub')
# API Key Input in the sidebar
api_key = st.sidebar.text_input("Enter your Google Gemini API Key:", type="password")

if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.sidebar.error(f"Error initializing Gemini: {e}")
        st.stop()
else:
    st.sidebar.warning("Please enter your Google Gemini API Key.")
    st.stop()
    
st.sidebar.title('Upload Excel or CSV File')

# Move the file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    # Validate the uploaded file
    required_columns = {'Disposition Date', 'Category', 'Disposition Name', 'Call'}
    if not required_columns.issubset(df.columns):
        st.error('Uploaded file does not have the required columns.')
        st.stop()

    st.sidebar.markdown('''
    ---
    Developed by the <span style="font-weight:bold; color:green;">Forecasting COE</span>
    ''', unsafe_allow_html=True)

    # Tabs
    tabs = st.tabs(["Monthly Dashboard","Weekly Dashboard"])

    # Data preprocessing
    df1 = df[['Disposition Date', 'Category', 'Disposition Name', 'Call']].copy()
    df1['Disposition Date'] = pd.to_datetime(df1['Disposition Date'])
    df1.rename(columns={'Disposition Date': 'Date', 'Disposition Name': 'Disposition_Name'}, inplace=True)

    df1['Year'] = df1['Date'].dt.year
    df1['Month'] = df1['Date'].dt.month
    df1['Day'] = df1['Date'].dt.day
    df1['Day_Name'] = df1['Date'].dt.strftime('%a')
    df1['Date_str'] = df1['Date'].dt.strftime('%Y-%m-%d')
    df1['Year_Week'] = df1['Date'].dt.strftime('%Y-%U')

    agg_df = df1.groupby(['Date_str', 'Year', 'Month', 'Day_Name', 'Year_Week', 'Category', 'Disposition_Name']).agg(call_count=('Call', 'count')).reset_index().sort_values(
        ['Year_Week', 'Date_str', 'Year', 'Month', 'Day_Name', 'Category'])

    # Calculate percentage distribution for each category
    category_distribution = agg_df.groupby('Category')['call_count'].sum().reset_index()
    category_distribution['percentage'] = (category_distribution['call_count'] / category_distribution['call_count'].sum()) * 100

    with tabs[0]:
        # --- Monthly Call Count by Category ---

        agg_df['Date'] = pd.to_datetime(agg_df['Date_str'])
        agg_df['Year_Month'] = agg_df['Date'].dt.to_period('M').astype(str)

        # Group by Year_Month and Category only once.
        monthly_calls = agg_df.groupby(['Year_Month', 'Category']).agg(
            call_count=('call_count', 'sum')).reset_index()

        # Ensure Year_Month is treated as a categorical type with order
        monthly_calls['Year_Month'] = pd.Categorical(monthly_calls['Year_Month'], categories=sorted(monthly_calls['Year_Month'].unique()), ordered=True)

        # --- Top 10 Dispositions Pie Chart ---

        # Sort the 'Year_Month' values in descending order
        sorted_year_month = sorted(monthly_calls['Year_Month'].unique(), reverse=True)

        selected_year_month_pie = st.selectbox('Select Year-Month for Dispositions:', sorted_year_month, index=0)
        if selected_year_month_pie:
            filtered_df_pie = agg_df[agg_df['Year_Month'] == selected_year_month_pie]
            top_10_dispositions_pie = filtered_df_pie.groupby('Disposition_Name')['call_count'].sum().nlargest(10).index
            filtered_df_pie = filtered_df_pie[filtered_df_pie['Disposition_Name'].isin(top_10_dispositions_pie)]

            table_data_pie = filtered_df_pie.groupby(['Year_Month', 'Category', 'Disposition_Name']).agg(
                call_count=('call_count', 'sum')).reset_index()

            table_data_pie = table_data_pie[table_data_pie['Disposition_Name'].isin(top_10_dispositions_pie)]
            table_data_pie = table_data_pie.sort_values(by='call_count', ascending=False).head(10)

            fig2 = px.pie(table_data_pie,
                          values='call_count',
                          names='Disposition_Name',
                          title=f'Top 10 Dispositions in {selected_year_month_pie}',
                          hole=0.3,
                          color_discrete_sequence=px.colors.qualitative.Pastel)

            fig2.update_traces(textposition='outside',
                               textinfo='label+percent',
                               textfont_size=10,
                               pull=[0.1] * len(table_data_pie))

            fig2.update_layout(
                uniformtext_minsize=10,
                uniformtext_mode='hide',
                title_x=0.3,
                width=800,
                height=600,
                margin=dict(t=80, b=80, l=80, r=80)
            )
            st.plotly_chart(fig2)

            st.markdown('<div class="center-title">Top 10 Dispositions</div>', unsafe_allow_html=True)
    
            st.dataframe(table_data_pie)

                    # --- Category Distribution Pie Chart ---

            fig3 = px.pie(category_distribution,
                          values='percentage',
                          names='Category',
                          title='Distribution of volumes per Channel',
                          color_discrete_sequence=px.colors.qualitative.Pastel)

            fig3.update_traces(textposition='outside',
                               textinfo='label+percent',
                               textfont_size=11,
                               pull=[0.1] * len(category_distribution))

            fig3.update_layout(
                uniformtext_minsize=10,
                uniformtext_mode='hide',
                title_x=0.35,
                width=800,
                height=600,
                margin=dict(t=80, b=80, l=80, r=80)
            )
            st.plotly_chart(fig3)
            
            fig1 = px.bar(monthly_calls,
                          x='Year_Month',
                          y='call_count',
                          color='Category',
                          title='Monthly Volume Count by Channel',
                          barmode='stack')

            fig1.update_layout(
                xaxis_tickangle=-45,
                legend_title='Category',
                width=800,
                height=600,
                title_x=0.3,
                margin=dict(t=80, b=80, l=80, r=80)
            )

            fig1.update_xaxes(
                tickformat="%Y-%m",
                tickvals=monthly_calls['Year_Month'],
                ticktext=monthly_calls['Year_Month'].astype(str)
            )

            st.plotly_chart(fig1)

            # --- Gemini 2.0 Flash Summary ---

            try:
                # Get top 5 dispositions
                top_5_dispositions = table_data_pie.nlargest(5, 'call_count')['Disposition_Name'].tolist()

                # Prepare data for Gemini (filter to the selected year-week)
                gemini_data = agg_df[agg_df['Year_Month'] == selected_year_month_pie].copy() 

                # Convert DataFrame to CSV string
                csv_string = gemini_data.to_csv(index=False)
                
                # Save DataFrame to CSV file
                gemini_data.to_csv('gemini_data.csv', index=False)

                # Save DataFrame to Excel file
                gemini_data.to_excel('gemini_data.xlsx', index=False)

                # Construct prompt for Gemini
                prompt = f"""
                Analyze the following call disposition data for the week of {selected_year_month_pie}. The data is provided in CSV format:

                {csv_string}

                The top 5 call dispositions, based on their frequency in the provided data, are: {', '.join(top_5_dispositions)}.

                For each of these top 5 dispositions, provide a summary based STRICTLY on the provided CSV or XLSX data that includes:

                1. **Occurrence:** Describe the total count of each disposition for {selected_year_month_pie} as it appears in the provided CSV data.
                2. **Distribution by Channel:** For each of the top 5 dispositions, provide the distribution of call counts across different categories (e.g., if there is a 'Category' column, show how many calls of each disposition fall into each category). If there is no category column then state "No category column available in data".

                Provide a concise and insightful summary of your findings for {selected_year_month_pie}, based STRICTLY on the provided CSV or XLSX data. Do not invent information.
                """


                # Generate summary using Gemini
                response = model.generate_content(prompt)
                summary = response.text

                # Display summary
                st.subheader("Summary")
                st.write(summary)

            except Exception as e:
                st.error(f"Error generating summary: {e}")
                
    with tabs[1]:
   
            weekly_view = agg_df.groupby(['Year_Week', 'Category']).agg(
            call_count=('call_count', 'sum')).reset_index()

            # Get top 5 categories for each week
            top_5_categories_weekly = weekly_view.groupby('Year_Week').apply(
                lambda x: x.nlargest(5, 'call_count')
            ).reset_index(drop=True)

            top_5_categories_weekly = top_5_categories_weekly.sort_values(by='Year_Week', ascending=False)

            # Convert 'Year_Week' to datetime format with week number
            weekly_view['Year_Week_dt'] = pd.to_datetime(weekly_view['Year_Week'] + '-1', format='%Y-%W-%w')

            categories = agg_df['Category'].unique().tolist()
            dispositions = agg_df['Disposition_Name'].unique().tolist()
            year_weeks = sorted(agg_df['Year_Week'].unique().tolist(), reverse=True)
            selected_year_week = st.selectbox('Select Year-week:', year_weeks, index=0)

            if selected_year_week:
                filtered_df_week = agg_df[agg_df['Year_Week'] == selected_year_week]
                top_10_dispositions_weekly = \
                    filtered_df_week.groupby('Disposition_Name')['call_count'].sum().nlargest(10).index
                filtered_df_week = filtered_df_week[filtered_df_week['Disposition_Name'].isin(top_10_dispositions_weekly)]

                chart_data_week = filtered_df_week.groupby('Disposition_Name')['call_count'].sum()
                table_data_week = filtered_df_week.groupby(
                    ['Year_Week', 'Category', 'Disposition_Name']).agg(
                    call_count=('call_count', 'sum')).reset_index()

                # Ensure only top 10 dispositions are shown
                table_data_week = table_data_week[table_data_week['Disposition_Name'].isin(top_10_dispositions_weekly)]
                table_data_week = table_data_week.sort_values(by='call_count', ascending=False).head(10)

                # Create the interactive pie chart with Plotly Express for Disposition Distribution
                fig2 = px.pie(table_data_week,
                                values='call_count',
                                names='Disposition_Name',
                                title=f'Disposition Distribution in {selected_year_week}',
                                hole=0.3,
                                color_discrete_sequence=px.colors.qualitative.Pastel)

                fig2.update_traces(textposition='outside',
                                    textinfo='label+percent',
                                    textfont_size=10,
                                    pull=[0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0])

                fig2.update_layout(uniformtext_minsize=10, uniformtext_mode='hide',
                                    title_x=0.3,
                                    width=800,
                                    height=600,
                                    margin=dict(t=80, b=80, l=80, r=80))
                st.plotly_chart(fig2)
                st.markdown('<div class="center-title">Top 10 Dispositions</div>', unsafe_allow_html=True)
                st.dataframe(table_data_week)


                # Create a pie chart for category distribution in the selected week.
                category_distribution = weekly_view[weekly_view['Year_Week'] == selected_year_week]

                if not category_distribution.empty:
                    fig_category = px.pie(category_distribution,
                                          values='call_count',
                                          names='Category',
                                          title=f'Channel Distribution in {selected_year_week}',hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)

                    fig_category.update_traces(textposition='outside',
                                        textinfo='label+percent',
                                        textfont_size=10,
                                        pull=[0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0])

                    fig_category.update_layout(uniformtext_minsize=10, uniformtext_mode='hide',
                                        title_x=0.3,
                                        width=800,
                                        height=600,
                                        margin=dict(t=80, b=80, l=80, r=80))
                    st.plotly_chart(fig_category)
                else:
                    st.write(f"No category data available for {selected_year_week}")

                fig = px.bar(weekly_view,
                                x='Year_Week_dt',
                                y='call_count',
                                color='Category',
                                title='Weekly Call Count by Category',
                                labels={'Year_Week_dt': 'Year-Week', 'call_count': 'Call Count'},
                                barmode='stack')

                fig.update_layout(xaxis={'categoryorder': 'total descending'},
                                    xaxis_tickangle=-45,
                                    legend_title_text='Category',
                                    plot_bgcolor='white', title_x=0.3)

                fig.update_xaxes(tickformat="%Y-%W")

                st.plotly_chart(fig)

                            # --- Gemini 2.0 Flash Summary ---

            try:
                # Get top 5 dispositions
                top_5_weekdispositions = table_data_week.nlargest(5, 'call_count')['Disposition_Name'].tolist()

                # Prepare data for Gemini (filter to the selected year-week)
                gemini_data = agg_df[agg_df['Year_Week'] == selected_year_week].copy() 

                # Convert DataFrame to CSV string
                csv_weekstring = gemini_data.to_csv(index=False)
                
                # Save DataFrame to CSV file
                gemini_data.to_csv('gemini_weekdata.csv', index=False)

                # Save DataFrame to Excel file
                gemini_data.to_excel('gemini_weekdata.xlsx', index=False)

                # Construct prompt for Gemini
                prompt = f"""
                Analyze the following call disposition data for the year-week of {selected_year_week}. The data is provided in CSV format:

                {csv_weekstring}

                The top 5 call dispositions, based on their frequency in the provided data, are: {', '.join(top_5_dispositions)}.

                For each of these top 5 dispositions, provide a summary based STRICTLY on the provided CSV or XLSX data that includes:

                1. **Occurrence:** Describe the total count of each disposition for year-week {selected_year_week} as it appears in the provided CSV or XLSX data.
                2. **Distribution by Channel:** For each of the top 5 dispositions, provide the distribution of call counts across different categories (e.g., if there is a 'Category' column, show how many calls of each disposition fall into each category). If there is no category column then state "No category column available in data".

                Provide a concise and insightful summary of your findings for {selected_year_week}, based STRICTLY on the provided CSV or XLSX data. Do not invent information.
                """

                # Generate summary using Gemini
                response = model.generate_content(prompt)
                summary = response.text

                # Display summary
                st.subheader("Summary")
                st.write(summary)

            except Exception as e:
                st.error(f"Error generating summary: {e}")
