# Context of the project
You work for an ESN. A client asks you to develop a web application for him.

Your client is a life coach, he helps people feel good about their daily lives. To follow the morale of his clients in two coaching sessions, he asks them to write a short text every day.

Some time ago, an influencer recommended this coach to his community, he has since been totally overwhelmed and had the idea of ​​a digital tool to automate his follow-up. (ppff the coaches ...)

So you need to build and publish:

a database to store your coach's information
a REST API with fastAPI (or other) to be able to interact with this database
a web application with streamlit (or other) used as a graphical interface.
What a client (of the coach) should be able to do:

add text to today's date
modify a text on today's date
read your text on today's date or any date
What the coach should be able to do:

add / remove / rename a customer.
add / delete / modify certain customer information.
get a list of all his clients and the information stored on him.
for a certain customer on a certain date get the text of a customer, his majority feeling, his wheel of emotions (% of each feeling)
For a certain client, a certain day, a certain week, a certain month or a certain year: recover the average sentiment wheel over the period
For a certain day, a certain week, a certain month, a certain year: recover the wheel of the average feelings of all of its customers over the period.
Tip: use the databases used during sentiment research to simulate past content.
