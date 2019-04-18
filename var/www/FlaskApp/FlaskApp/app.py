import csv
from flask import Flask, render_template, request, redirect, url_for


app = Flask(__name__)

def generate_recommendation():
    with open("sample_output.csv") as csvfile:
        reader = csv.reader(csvfile)

        # a list to store all the movie names
        column_header = next(reader)

        # a matrix to store the binary data for users and the corresponding movies
        binary_data = [row for row in reader]
        # print(column_header)
        # print(binary_data)

        # recommendatio list
        rec_list = []

        # temporary list that will be appended to to rec_list after it is updated for each user
        temp_list = []

        # for loop for users i.e. rows
        for i in range(0, 9):
            temp_list.append(i + 1)

            # for loop for movies i.e. columns
            for j in range(1, 7):
                if binary_data[i][j] == '1':
                    temp_list.append(column_header[j])
            rec_list.append(temp_list)
            temp_list = []

        print(rec_list)

    return rec_list


# app = Flask(__name__)
#
# @app.before_request
# def before_request():
#     # create a db if needed and connect
#     initialize_db()
#
# @app.tear_down
# def tear_down(exception):
#     # close the db connection
#     db.close()

@app.route('/')
def home():
    # render the home page with the recommendations
    final_rec_list = generate_recommendation()
    return render_template('index4.html', final_rec_list=final_rec_list)


if __name__ == '__main__':
    app.run(debug = True)






