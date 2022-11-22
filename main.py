import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from diet_recommendations import Weight_Loss, Weight_Gain
import sklearn
import pickle

logistic_regression_model = pickle.load(open('logistic_regression.pkl', 'rb'))
nav_model = pickle.load(open('naive_bayes.pkl', 'rb'))
svm = pickle.load(open('svm.pkl', 'rb'))


def classify(num):
    if num == [[0, 0]]:
        return 'Average Health and Financial well-being'
    elif num == [[0, 1]]:
        return 'Average Health and Good Financial well-being'
    elif num == [[0, 2]]:
        return 'Average Health and Low Financial well-being'
    elif num == [[1, 1]]:
        return 'Low health and good Financial well-being'
    elif num == [[1, 0]]:
        return 'Low health and Average Financial well-being'
    elif num == [[1, 2]]:
        return 'Low health and Bad Financial well-being'
    elif num == [[2, 0]]:
        return 'Good health and Average Financial well-being'
    elif num == [[2, 1]]:
        return 'Good health and Good Financial well-being'
    else:
        return 'Good health and Low Financial well-being'


def main():
    st.title("Panacea Behaviour Analysis")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Health Financial Score Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities = ['Logistic Regression', 'Naive Bayes', 'SVM']
    option = st.sidebar.selectbox('Which model would you like to use?', activities)
    st.subheader(option)
    age = st.slider('How old are you?', 0, 50, 135)
    Gender = st.radio('Select your Gender', ('Female', 'Male'))
    if Gender == 'Female':
        gender = 1
    else:
        gender = 2

    height = st.number_input('Enter the height in cm', value=1)
    weight = st.number_input('Enter the weight in kg', value=1)
    bmi = int(weight / (height / 100) ** 2)

    fruit = st.radio('HOW MANY FRUITS OR VEGETABLES DO YOU EAT EVERYDAY? In a typical day, averaging workdays and '
                     'weekends.', ('less than 2', 'exactly 2', 'between 3 and 4', 'between 5 and 6', 'more than 6 '))
    if fruit == 'less than 2':
        fruits = 1
    elif fruit == 'exactly 2':
        fruits = 2
    elif fruit == 'between 3 and 4':
        fruits = 3
    elif fruit == 'between 5 and 6':
        fruits = 4
    else:
        fruits = 5

    # st.write('The current number is ', fruits)

    meditation = st.radio('IN A TYPICAL WEEK, HOW MANY TIMES DO YOU HAVE THE OPPORTUNITY TO THINK ABOUT YOURSELF? '
                          'Include meditation, praying and relaxation activities such as fitness, walking in a park '
                          'or lunch breaks.',
                          ('Not an single day in a week', 'Not frequent on alternate days', 'Daily'))
    if meditation == 'Not an single day in a week':
        meditation = 1
    elif meditation == 'Not frequent on alternate days':
        meditation = 2
    else:
        meditation = 5

    sleep_hour = st.radio('ABOUT HOW LONG DO YOU TYPICALLY SLEEP?',
                          ('less than 4hrs', 'between 4hr to 6hrs', 'between 6hr '
                                                                    'to 7hr',
                           'between 7hr to 8 hr', 'more than 8hr'))
    if sleep_hour == 'less than 4hrs':
        sleep = 1
    elif sleep_hour == 'between 4hr to 6hrs':
        sleep = 2
    elif sleep_hour == 'between 6hr to 7hr':
        sleep = 3
    elif sleep_hour == 'between 7hr to 8 hr':
        sleep = 4
    else:
        sleep = 5

    daily_steps = st.radio('HOW MANY STEPS (IN THOUSANDS) DO YOU TYPICALLY WALK EVERYDAY?', (
        'less than 4000', 'around 5000 per day', 'around 5000 to 8000', 'around 8000 to 10,000', 'more than 10,000'))

    if daily_steps == 'less than 4000':
        steps = 1
    elif sleep_hour == 'around 5000 per day':
        steps = 2
    elif sleep_hour == 'around 5000 to 8000':
        steps = 3
    elif sleep_hour == 'around 8000 to 10,000':
        steps = 4
    else:
        steps = 5

    handle_expense = st.radio('I could handle a major unexpected expense',
                              ('Not at all', 'very little', 'somewhat', 'very well', 'completely'))
    if handle_expense == 'Not at all':
        handle_expenses = 1
    elif handle_expense == 'very little':
        handle_expenses = 2
    elif handle_expense == 'somewhat':
        handle_expenses = 3
    elif handle_expense == 'very well':
        handle_expenses = 4
    else:
        handle_expenses = 5

    savehabits = st.radio('Putting money into savings is a habit for me',
                          ('Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'))
    if savehabits == 'Strongly Disagree':
        savehabit = 1
    elif savehabits == 'Disagree':
        savehabit = 2
    elif savehabits == 'Neutral':
        savehabit = 3
    elif savehabits == 'Agree':
        savehabit = 4
    else:
        savehabit = 5

    follow_financial_goal = st.radio('I follow-through on financial goals I set for myself', ('Not at all', 'very '
                                                                                                            'little',
                                                                                              'somewhat', 'very well',
                                                                                              'completely'))
    if follow_financial_goal == 'Not at all':
        follow_financial_goals = 1
    elif follow_financial_goal == 'very little':
        follow_financial_goals = 2
    elif follow_financial_goal == 'somewhat':
        follow_financial_goals = 3
    elif follow_financial_goal == 'very well':
        follow_financial_goals = 4
    else:
        follow_financial_goals = 5

    financial_planning_time_horizons = st.radio('Financial planning time horizon', (
        'The next few months', 'The next year', 'The next few years', 'The next 5 to 10'
                                                                      'years', 'Longer than 10 years '))
    if financial_planning_time_horizons == 'Not at all':
        financial_planning_time_horizon = 1
    elif financial_planning_time_horizons == 'very little':
        financial_planning_time_horizon = 2
    elif financial_planning_time_horizons == 'somewhat':
        financial_planning_time_horizon = 3
    elif financial_planning_time_horizons == 'very well':
        financial_planning_time_horizon = 4
    else:
        financial_planning_time_horizon = 5

    mean_health = (((fruits + meditation + sleep + steps) / 4) * 10)
    financial_mean = (
                ((handle_expenses + savehabit + follow_financial_goals + financial_planning_time_horizon) / 4) * 10)
    inputs = [[age, gender, bmi, fruits, meditation, sleep, steps, handle_expenses, savehabit, follow_financial_goals,
               financial_planning_time_horizon]]

    if st.button('Submit'):
        if option == 'Logistic Regression':
            predicted_values = logistic_regression_model.predict(inputs)
            predicted_value = np.array(predicted_values).tolist()
            plot(bmi, mean_health, financial_mean)
            st.success(classify(predicted_value))
            st.header("Recommended Food items for Breakfast")
            recommendation_plot(age, bmi)

        elif option == 'Naive Bayes':
            predicted_values = nav_model.predict(inputs)
            predicted_value = np.array(predicted_values).tolist()
            st.pyplot(plot(bmi, mean_health, financial_mean))
            st.success(classify(predicted_value))
        else:
            predicted_values = svm.predict(inputs)
            predicted_value = np.array(predicted_values).tolist()
            st.pyplot(plot(bmi, mean_health, financial_mean))
            st.success(classify(predicted_value))
        # print(bmi,sleep,gender,predicted_value)


def plot(bmi, mean_health, financial_mean):
    if bmi < 18:
        labels = 'Underweight', 'Health Score', 'Financial Score'
    elif 18 <= bmi < 25:
        labels = 'Normal weight', 'Health Score', 'Financial Score'
    elif 25 <= bmi < 30:
        labels = 'Overweight', 'Health Score', 'Financial Score'
    else:
        labels = 'Obesity', 'Health Score', 'Financial Score'

    sizes = [bmi, mean_health, financial_mean]

    my_circle = plt.Circle((0, 0), 0.7, color='white')
    plt.pie(sizes, labels=labels, autopct="%.1f%%", wedgeprops={'linewidth': 7, 'edgecolor': 'white'})
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    return st.pyplot(p)


def recommendation_plot(age, bmis):
    t1 = Weight_Loss(age, bmis, 1)
    labels = t1[0]
    print(labels)
    size = t1[1]
    print(size)
    # explode = (0.1, 0, 0)
    fig1, ax1 = plt.subplots()
    ax1.pie(size,  labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    return st.pyplot(fig1)


if __name__ == '__main__':
    main()
