import streamlit as st
import pandas as pd

rest_win_subcategory = ['Beach Bars', 'Mexican', 'Bavarian', 'Cafeterias', 'Himalayan/Nepalese']
other_win_subcategory = 'Zoo'
count_businesses = 755


st.title('Investment Analysis for Businesses in Berlin - using Yelp.com')

st.write('How many businesses did we take into consideration? *{count_businesses}*'.format(count_businesses=count_businesses))
st.subheader('We will present the best option for each main sub-category')

data_frame1 = pd.read_csv('data/shit.csv')
data_frame2 = pd.read_csv('data/other.csv')
data_frame_corr = pd.read_csv('data/insignificant_correlations.csv')

if st.button("For Restaurants"):
    st.subheader('Winning sub-category: ')
    st.write('{rest_win_subcategory}'.format(rest_win_subcategory=rest_win_subcategory))
    st.write(data_frame1)
    st.subheader('Key Factors for success: ')
    st.write('There was no correlation')
    st.write(data_frame_corr)
    # st.write('Days Open: {rest_days}'.format(rest_days=rest_days))
    # st.write('Days Open: {rest_att}'.format(rest_att=rest_att))


if st.button("For Other Business Types"):
    st.subheader('Winning sub-category: ')
    st.write("We can see that generally customers are happy with the service but if we could choose one we would choose a Zoo because it's the worst rated.")
    st.write('{other_win_subcategory}'.format(other_win_subcategory=other_win_subcategory))
    st.write(data_frame2)
    st.subheader('Key Factors for success: ')
    st.write('Not enough data to take significant conclusion')


if st.button("Assumptions"):
    st.write("""Using the Pareto Law we looked for:
    
    + The worst rated categories - High Opportunity of Improvement (unhappy clients)

    + The categories with the most reviews - High Demand

    + The categories with the least businesses - Low Supply
    
    """)
