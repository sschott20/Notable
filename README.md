# Notable

## Inspiration

Many studies have shown that taking notes by hand increases material retention. But it also increases something else--the chance of losing your work. What if you could have the learning benefits of handwriting notes but still be able to keep a copy as a Google or Word document and Ctrl-F through it later? As two students who spent the past year studying machine learning, we knew we had to create our own solution.

## What it does

Notable is a web app that allows you to input pictures of your notes and have it transcribed for you into a text document of your choice using our deep learning model.

## How we built it

We built our app with a 4-tier architecture integrated into both the cloud and the browser. We aggregated data from the IAM Handwriting Database, the Bentham Manuscripts Collection, the RIMES Letter Database, and the Saint Gall Database and trained our model on Google Cloud Platform’s Cloud ML Engine. We then served our model with Docker and Flask in an easy to use, web application.

Our model's training can be divided into three steps. First, our preprocessed images are fed into a five-layer convolutional neural network to extract features. Next, the outputed feature map is propagated through a Long Short Term Memory Network. Finally, we use CTC to both calculate the loss for the RMSProp optimizer as well as decode into our final text.

## Challenges we ran into

After bricking our computers trying to download all the data, we decided to move our data aggregation and model training to Google Cloud Platform’s Cloud ML Engine. This allowed us much more time for optimizing our model and creating our Flask interface. Also, we spent much more time than we expected preprocessing our data.

## Accomplishments that we are proud of

Figuring out how to integrate Google Cloud Platform into our workflow was a lifesaver. Our app would not be where it is without it.

## What we learned

We learned a ton about Convolutional Neural Networks and Long Short Term Memory Networks while building our project, as well as integrating machine learning with flask to create an easy to use and nice looking ui instead of a command line.

## What's next for Notable

There's still room to improve our model through more data and better architecture, which is going to be vital going forward. We also have plenty of work to do in making our quick hackathon web app into a full-fledged application/website.
