import csv
import random
from flask import Flask, render_template, request, redirect, url_for

number_of_users = 100
number_of_movies = 100


app = Flask(__name__)


def random_colour_generation():
	colour = ['aqua', 'green', 'yellow', 'red']
	secure_random = random.SystemRandom()
	return secure_random.choice(colour)


def generate_recommendation():
    with open("output_new.csv") as csvfile:
        reader = csv.reader(csvfile)

        # a list to store all the movie names
        column_header = next(reader)

        # a matrix to store the binary data for users and the corresponding movies
        binary_data = [row for row in reader]
        # print(column_header)
        # print(binary_data)

        # recommendation list
        rec_list = []

        # temporary list that will be appended to to rec_list after it is updated for each user
        temp_list = []

        # for loop for users i.e. rows
        for i in range(0, number_of_users):
            temp_list.append(i + 1)

            # for loop for movies i.e. columns
            for j in range(1, number_of_movies):
                if binary_data[i][j] == '1':
                    temp_list.append(column_header[j])
            rec_list.append(temp_list)
            temp_list = []

        print(rec_list)

    return rec_list


@app.route('/')
@app.route('/index/')
@app.route('/home/')
def index():
		return render_template('index_form.html')
		
@app.route('/check_user_id/', methods=['POST'])
def check_user_id():

	# function to render the page only with user specific recommendations
	user_id = int(request.form['user_id'])
	number_of_movies = int(request.form['no_of_movies'])
	final_rec_list = generate_recommendation()
	
	# user specific recommendation list is present in serial order of the user_id
	if user_id >= 1:
		user_rec_list = final_rec_list[user_id-1]
	
	return render_template('user_recommendation_page.html', user_rec_list = user_rec_list, colour = random_colour_generation(), user_id = user_id, number_of_movies = number_of_movies) 
	
	
	
	



@app.route('/recommendations_page/')
def recommendations_page():
    # function to render the page with all the recommendations
    final_rec_list = generate_recommendation()
	
	# change the for loop range which generates the small box for movies depending upon the number of movies
    return render_template('recommendation_page.html', final_rec_list = final_rec_list, colour = random_colour_generation())
	
	
	



if __name__ == '__main__':
    app.run(debug = True)






