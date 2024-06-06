from flask import Flask, render_template, url_for
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os
import sqlite3

app = Flask(__name__)

DATABASE = 'data.db'

# for matplotlib to work in flask
import matplotlib
matplotlib.use('Agg')

if not os.path.isdir('static'):
    os.mkdir('static')

def init_db():
    # Initialize SQLite database and create tables if they don't exist
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sales_data (
                OrderDate TEXT,
                Sales REAL,
                OrderCount INTEGER,
                OrderQuantity INTEGER,
                YEAR_MONTH TEXT
            )
        ''')
        conn.commit()

@app.route('/')
def index():
    paths = []
    # Extract
    df = pd.read_excel('/Path/to/file.xlsx')

    # Add a 'YEAR_MONTH' column by extracting year and month from 'OrderDate'
    df['YEAR_MONTH'] = df['OrderDate'].apply(lambda x: x.strftime('%Y-%m'))

    # lets print some KPIs
    KPIs = (
        'Total Sales: '+ str('{:,.0f}'.format(round(df.Sales.sum()/1000))),
        'Total Order Count: ' + str('{:,.0f}'.format(round(df.OrderCount.sum()/1000))),
        'Total Order Quantity: ' + str('{:,.0f}'.format(round(df.OrderQuantity.sum()/1000))),
    )

    ######################### FIRST TRANSFORMATION (MONTHLY SALES) ###########################

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    result = df.groupby('YEAR_MONTH')[numeric_cols].sum()

    # Load transformed data in SQLite database
    with sqlite3.connect(DATABASE) as conn:
        df.to_sql('original_sales_data', conn, if_exists='replace', index=False)

    months = [month for month, _ in df.groupby('YEAR_MONTH')]

    fig = plt.figure(figsize=(16, 7))

    plt.plot(months, result['Sales'], color='#800045')
    plt.xticks(months, rotation='vertical', size=8)
    plt.ylabel('Sales in USD')
    plt.xlabel('Months')

    imagepath = os.path.join('static', 'Sales_month.png')
    paths.append(imagepath)

    # base64 encoding of image plot
    Sales_month_img = processImg(fig)
    
    # Save PNG for plot in static(optional as we already base64encoding img to index.html)
    plt.savefig(imagepath)
    plt.close()

    
    ######################### SECOND TRANSFORMATION (Sales FOR PreviousYear) ###########################

    pd.options.display.float_format = '{:,.2f}'.format
    prod_sales_sum = df.groupby('product')[numeric_cols].sum()
    prod_sales_sum.sort_values(by=['Sales'], inplace=True, ascending=False)
    # top_10 = prod_sales_sum.head(10)

    ndf = df[['OrderDate','Sales']]
    new_df = ndf.groupby('OrderDate')['Sales'].sum().reset_index()
    new_df = new_df.drop_duplicates('OrderDate').set_index('OrderDate').asfreq('D', fill_value=0)
    new_df = new_df.sort_index().reset_index()

    # Calculate sales for the previous year on the same date
    new_df['Previous'] = new_df.groupby([new_df['OrderDate'].dt.month, new_df['OrderDate'].dt.day])['Sales'].shift()
    
    # Aggregate summed data by year-month
    new_df['YEAR_MONTH'] = new_df['OrderDate'].apply(lambda x: x.strftime('%Y-%m'))
    new_df = new_df.groupby('YEAR_MONTH')[['Sales','Previous']].sum()

    # new_df.plot(kind='bar', figsize=(15,5)) other way to plot (simple way)
    
    # set style for plotting
    plt.style.use('ggplot')

    # Create bar chart
    # positions = range(len(new_df))  # Positions for the bars
    fig = plt.figure(figsize=(16,8))
    positions = np.arange(len(new_df))  # Array of positions: [0, 1, 2, ..., N-1]
    # Width of each bar. Smaller value than 0.5 allows space between groups
    bar_width = 0.4
    # Plotting 'Sales'
    plt.bar(positions - bar_width / 2, new_df['Sales'], width=bar_width, color='red', label='Sales')
    # Plotting 'Previous' right next to 'Sales'
    plt.bar(positions + (bar_width / 2)+0.04, new_df['Previous'], width=bar_width, color='green', label='Previous')
    # Adding labels, title, and legend
    plt.xticks(positions, new_df.index, rotation='vertical')  # Use the index for labels if 'year_month' is not a separate variable
    plt.xlabel('Year-Month')
    plt.ylabel('Amount')
    plt.title('Comparison of Sales and Previous Period')
    plt.legend()

    imagepath = os.path.join('static', 'Sales_PreviousYear.png')
    paths.append(imagepath)

    # Save PNG for plot in static(optional as we already base64encoding img to index.html)
    plt.savefig(imagepath)
    plt.close()
    # base64 encoding of img plot
    Sales_PreviousYear = processImg(fig)


    ######################### THIRD TRANSFORMATION (CUMSUM) ###########################

    df_chart = df.groupby(['Country', 'productcategory'])['Sales'].sum().reset_index()

    df_cumsum = df_chart.groupby('Country')['Sales'].sum().reset_index()
    df_cumsum = df_cumsum.rename(columns={'Sales':'Total_sales'})
    df_combined = pd.merge(df_chart, df_cumsum, on='Country', how='left')
    df_combined['Sales_per'] = df_combined['Sales']/df_combined['Total_sales']
    df_combined.drop(['Sales','Total_sales'], axis=1, inplace=True)
    df_pivoted = pd.pivot_table(df_combined, values='Sales_per', index=['Country'], columns='productcategory').reset_index()
    df_pivoted.plot(
        x='Country',
        kind='barh',
        stacked=True,
        figsize=(15,5),
        color=['orange', 'blue', 'green', 'purple'],
        title='Sales distribution by category',
        mark_right=True
    )
    plt.legend(bbox_to_anchor=(1.0,1.0))

    # code to show percentage on the bars
    df_total = df_pivoted["Accessories"] + df_pivoted["Bikes"] + df_pivoted["Clothing"] + df_pivoted['Components']
    df_rel = df_pivoted[df_pivoted.columns[1:]].div(df_total, 0)*100
    
    for n in df_rel: 
        for i, (cs, ab, pc) in enumerate(zip(df_pivoted.iloc[:, 1:].cumsum(1)[n],  
                                            df_pivoted[n], df_rel[n])): 
            plt.text(cs - ab / 2, i, str(np.round(pc, 1)) + '%',  
                    va = 'center', ha = 'center')
            
    imagepath = os.path.join('static', 'SalesByCategory.png')
    # Save PNG for plot in static( as we are Not base64encoding for this img)
    plt.savefig(imagepath)

    with sqlite3.connect(DATABASE) as conn:
        df_pivoted.to_sql('Transformed_data(cumsum)', conn, if_exists='replace', index=False)

    return render_template('index.html', Sales_month_img=Sales_month_img, Sales_PreviousYear=Sales_PreviousYear, img3='SalesByCategory.png', KPIs=KPIs)


def processImg(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

if __name__ == '__main__':
    app.run(debug=True)
