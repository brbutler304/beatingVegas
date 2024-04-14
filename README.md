# beatingVegas

Brian R Butler - Code KY (DA 2) July 2023

Thank you for taking the time to review my project. Baseball is one of my biggest passions, especially when it is combined with data analysis! I've spent countless hours wrangling data and developing a few models to aid in the prediction of outcomes for MLB games - I sincerely hope you enjoy it, learn something, and find it as useful as I did when building the models.

Due to run time, I have supplied you with all three (3) directories created by my code (batter_data, oddshark, and SP_data) and will comment out the codeblocks which reproduce it. I promise that the code works and I encourage you to run them but be prepared to walk away for 8 hours or so (I recommend >8GB RAM but 8GB RAM will process it - albeit, slowly).

I also want to explicitly state that due to Retrosheet's incredible cataloging of data, we, in theory, could build a model from data as far back as 1871 when baseball began. However, "The Modern Era" of baseball as we know it today began in 1901. The first time that I built this model, I thought it would be really neat to see how training this model with data as far back as 1901 would compare to say, starting in 1950. To my surprise, the models compared relatively well but I suspect that was largely due to there being so many missing values in the data from the early decades. So, to save on memory, we will begin our model in 1970.

nb 1 - Data Wrangling

    In this notebook, we will wrangle data downloaded from www.retrosheet.org into a dataframe suitable for model building. Specifically, for each game, we will calculate some team statistics over their past 162 games. At the end, we save our dataframe to a file. This file will be the starting point for the next notebook, in which we build our first model. To use this notebook, you must first download the game logs here: https://www.retrosheet.org/gamelogs/index.html. Towards the bottom of the page there are links for ZIP files containing multiple seasons. Download the zip files containing the 'gamelog' data, decompress it, and then move all of the single season files to a single directory. You will then need the path to that directory for the variable fname. Once all dataframes are concatenated, we will save it to a .csv file for later use.

nb 2 - Building our First Model

    In the last notebook, we loaded in the data from retrosheet and did some processing to get the team level statistics. We saved that data to a file called 'df_bp1.csv'. In this notebook, we will load in that data, do some initital data exploration, and then build and evaluate our first predictive model.

nb 3a - Getting Odds Data

    In this notebook, we will get historical odds data from oddsshark.com (PROVIDED: /your/path/to/oddshark). We will use the pandas read_html function to grab a table into a dataframe, and show how to programmatically sweep through all the necessary urls to get the data we need. We will save this data as a collection of csv files. In the next notebook, we will use these csv files to add the odds information to our primary dataframe.

nb 3b - Augment df with Odds Data

    In the previous notebook, we got historical odds data from oddsshark.com and saved them as a set of csv files (with a particular naming convention). In this notebook we will load that data and augment our primary (game-level) data frame so that it includes this odds data - specifically, the implied probabilities and the over/under, for each game.

nb 4 - Analyze Odds Data

    In the last notebook, we obtained historical odds data from oddsshark, and then augmented our game level data to include the implied probabilities, and over/under lines. We saved that data to a file called 'df_bp3.csv' and in this notebook, we will do some initial exploration of that odds data, and compare the quality of our first model predictions to the implied probabilities given by the oddsmakers.

nb 5a - Scraping (Raw) Pitching Data

    In the previous notebook, we compared our simple, hitting-only model to the Las Vegas odds. We concluded that incorporating the starting pitcher information would be a crucial next step to improve our model. In this notebook we will learn how to scrape individual, game-level, pitching data from retrosheet. We will write a loop to go through and download the data. This will enable us to augment our game-level dataframe with features derived from the previous performance of the starting pitcher. Let's start by going to retrosheet and finding the stats for CC Sabathia (one of my favorite pitchers from my childhood) - specifically his 2007 Cy Young Award winning season!

nb 5b - Add Pitching Features

     Now that we have raw game-level data for each pitcher, we can derive features based on the starting pitchers to help our prediction model for individual games. For each starting pitcher we will load their raw data, create features for each game based on their previous performance, and then save the dataframe in a dictionary structure for easy lookup. Then we can iterate through our game-level dataframe, add in the features for each starting pitcher, and use those to improve our model.

nb 6 - New Model with SP Features Included

    In the last notebook we augmented our dataframe to include various features based on the starting pitcher's performance
    Now we will add these features in to see how much improvement we get to our model.

nb 7 - Add Bullpen Data

    - Bullpen = relief pitchers (i.e. a pitcher who enters the game who wasn't the starting pitcher [[INCLUDES CLOSER]])
    - Bullpen dynamics are very complicated! For simplicity, we will just consider the 'team bullpen' rather than individual players
    - For each game, we can look at the performance of the bullpen by subtracting the SP from the pitching stats.
    - Then, do 'n' game lookbacks (similar to how we did team hitting data) to create features based on recent bullpen performance
    - This will not, however, account for which pitchers are rested/available for the game in question

nb 8 - Model with Bullpen

    In the last notebook we augmented our dataframe to include various features based on the starting pitcher's performance. Now we will add these features in to see how much improvement we get to our model.

nb 9a - Scraping Raw Batter Data

    - Currently the model averages the recent team hitting performance
    - This does not account for the particular players in the starting lineup that day
    - e.g. If a key hitter is resting, injured, got traded, etc.
    - To begin to model this we first need to scrape the raw batter data (similar to how we got the pitching data)

nb 9b - Process Batter Data

    - Now, we need to process this data and augment our game-level data frame
    - For each game, we want the trailing statistics for each player designated in the starting lineup
    - Need to add the statistics for those 18 players to each game
    - From that, we may want to derive "team-level" averages to simplify our feature set
    - NOTE: this will complicate things when we "go into production" and try and use this model for predicting new games

nb 10 - Model with Lineup

    - Last time we scraped individual batter data for the 7300+ players who have appeared in a starting lineup between 1970 and 2022.
    - We then processed the data to get statistics about the trailing performance of each of the players before each game in which they started.
    - Using this, we were able to get features related to each player that could be used to predict each game. We then averaged across the lineup in several different ways to come up with a variety of lineup related statistics.
    - Now, we will add these features to the model and see how much improvement we get

nb 11a - Predicting Runs Scored

    Last notebook we decided to switch our focus onto predicting runs scored
    
    This could be useful in several capacities:
    - For predicting the over/under
    - To create features for predicting the game winner

    We will focus on predicting the distribution of runs scored - that is, putting a probability on each possible value of runs scored for a team (up to some maximum value)

    Predicting the distribution of a numeric target variable is known as "probabilistic regression"

    In this notebook, we will "wrangle" our data such that each row represents a hitting team against a pitching team
    The goal will be to predict the (distribution of the) number of runs scored by the hitting team

nb 11b - Model Runs Scored (First Model)

    - We will now use the data frame from before to build a model to predict the distribution of runs scored

    - We will use an algorithm called Coarsage to model these distributions (Coarsage is in the StructureBoost package)

    - Coarsage is similar to PrestoBoost (a paper on PrestoBoost can be found here: https://arxiv.org/abs/2210.16247)

    - But it is a bit cleaner (needs only a single forest)

nb 12 - Predicting Over/Under

    - Last time we built a model to predict runs scored given features about the hitting team and the opposing pitchers

    - In this notebook, we will use that model to predict the probability that the total runs scored is over or under the "over/under line" for each game in the test set

    - This first attempt involves an independence assumption which is likely not true in practice, but may be close enough to true to give good predictions

    - We will then evaluate how this model would have fared in practice had we made bets on it, and do some further analysis to see if we really have an edge on the Vegas probabilities for the over/under

nb 13 - Evaluating Our Edge

     - In this notebook, we repeat the evaluation of the over/under model but with a larger test set

     - We will then do some simulations to see what kinds of profit we could expect (assuming our edge is real)

nb 14 - Calibrate Runs Scored

     - In the last couple of notebooks, we built a model to predict the distribution of runs scored, given a hitting team and a pitching team

    - We then used an independence assumption to predict the probability that the number of runs scored was over, under, or equal to the Over/Under "line" for that game.

    - We were able to build a model that looked like it could profitably bet against the Las Vegas odds!

    - However, these final predictions were clearly not well-calibrated.

    - For example, there were 362 games where we expected to be right more than 70% of the time (ignoring pushes), but we were actually right only 56.7% of the time (ignoring pushes) (194/(194+148))

    - There are two possible sources for this error (could be either or both)

        -- The runs scored model is poorly calibrated

        -- The independence assumption is flawed in practice

    - In this notebook, we will explore the calibration of the runs scored model and scope possible solutions

nb 15 - Building a Nested Model for O/U

    - Last time, we explored the calibration of the individual team runs scored models

    - We observed some degree of mis-calibration: over-predicting the probability of low runs scored, under-predicting the probability of high runs scored

    - However, these discrepancies did not seem large enough to explain the high degree of miscalibration in the overall model

    - Concluded that the independence assumption must be flawed

    - In this notebook, we use nested modeling to predict the total runs scored
