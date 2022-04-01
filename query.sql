-- View the average goal for different outcomes
SELECT AVG(goal) AS average_goal, outcome FROM campaign 
GROUP BY outcome;

-- an overview of key parameters according to the outcome
SELECT outcome, COUNT(*) AS total_campaign, SUM(goal) AS total_goal, SUM(pledged) AS total_pledged, SUM(backers) AS total_backers
FROM campaign
GROUP BY outcome;

SELECT AVG(goal) AS average_goal, AVG(pledged) AS average_pledged, AVG(backers) AS average_backers
FROM campaign
WHERE (sub_category_id = 14 or sub_category_id = 70) AND outcome = 'successful';

--  check if the time frame of the campaign affect the amount of pledged money
SELECT DATEDIFF(deadline,launched) AS duration_days, AVG(goal), AVG(pledged), AVG(backers)
FROM campaign
WHERE NOT outcome = 'undefine' OR outcome = 'live'
GROUP BY duration_days
ORDER BY duration_days DESC;

-- check the maximum for each key variable
SELECT MAX(goal), MAX(pledged), MAX(backers) FROM campaign;
SELECT MIN(goal), MIN(pledged), MIN(backers) FROM campaign;

-- number of campaign
SELECT COUNT(*) FROM campaign;

-- Data Cleaning
CREATE TABLE cleaned_campaign AS
SELECT *, DATEDIFF(deadline,launched) AS duration_days FROM campaign 
WHERE (outcome = 'successful' OR outcome = 'failed') AND (launched >= '2010-01-01' AND launched < '2018-01-01') AND (deadline >= '2010-01-01' AND deadline < '2018-01-01');

-- Q1: Are the goals for dollars raised significantly different between campaigns that are successful and unsuccessful?
SELECT AVG(goal), outcome 
FROM cleaned_campaign
WHERE outcome = 'failed' OR outcome = 'successful'
GROUP BY outcome;

-- Q2: What are the top/bottom 3 categories with the most backers? What are the top/bottom 3 subcategories by backers?

-- Top3 categories by backers
SELECT SUM(c.backers) AS total_backers, category.name
FROM cleaned_campaign AS c
Join sub_category 
	ON c.sub_category_id = sub_category.id
JOIN category
	ON sub_category.category_id = category.id
GROUP BY category.name
ORDER BY total_backers DESC
LIMIT 3;
-- Bottom3 categories by backers
SELECT SUM(c.backers) AS total_backers, category.name
FROM cleaned_campaign AS c
Join sub_category 
	ON c.sub_category_id = sub_category.id
JOIN category
	ON sub_category.category_id = category.id
GROUP BY category.name
ORDER BY total_backers ASC
LIMIT 3;

-- Top3 subcategories by backers
SELECT SUM(c.backers) AS total_backers, sub_category.name
FROM cleaned_campaign AS c
Join sub_category 
	ON c.sub_category_id = sub_category.id
JOIN category
	ON sub_category.category_id = category.id
GROUP BY sub_category.name
ORDER BY total_backers DESC
LIMIT 3;
-- Bottom3 subcategories by backers
SELECT SUM(c.backers) AS total_backers, sub_category.name
FROM cleaned_campaign AS c
Join sub_category 
	ON c.sub_category_id = sub_category.id
JOIN category
	ON sub_category.category_id = category.id
GROUP BY sub_category.name
ORDER BY total_backers ASC
LIMIT 3;


-- Q3: What are the top/bottom 3 categories that have raised the most money? What are the top/bottom 3 subcategories that have raised the most money?

-- Top3 categories by amount pledged
SELECT SUM(c.pledged) AS total_pledged, category.name
FROM cleaned_campaign AS c
Join sub_category 
	ON c.sub_category_id = sub_category.id
JOIN category
	ON sub_category.category_id = category.id
GROUP BY category.name
ORDER BY total_pledged DESC
LIMIT 3;
-- Bottom3 categories by amount pledged
SELECT SUM(c.pledged) AS total_pledged, category.name
FROM cleaned_campaign AS c
Join sub_category 
	ON c.sub_category_id = sub_category.id
JOIN category
	ON sub_category.category_id = category.id
GROUP BY category.name
ORDER BY total_pledged ASC
LIMIT 3;

-- Top3 subcategories by amount pledged
SELECT SUM(c.pledged) AS total_pledged, sub_category.name
FROM cleaned_campaign AS c
Join sub_category 
	ON c.sub_category_id = sub_category.id
JOIN category
	ON sub_category.category_id = category.id
GROUP BY sub_category.name
ORDER BY total_pledged DESC
LIMIT 3;
-- Bottom3 subcategories by amount pledged
SELECT SUM(c.pledged) AS total_pledged, sub_category.name
FROM cleaned_campaign AS c
Join sub_category 
	ON c.sub_category_id = sub_category.id
JOIN category
	ON sub_category.category_id = category.id
GROUP BY sub_category.name
ORDER BY total_pledged ASC
LIMIT 3;


-- Q4: What was the amount the most successful board game company raised? How many backers did they have?
SELECT id, name, goal, pledged, backers FROM cleaned_campaign
WHERE sub_category_id = 14 AND outcome = 'successful'  -- sub_category_id = 14 is Tabletop Games subcategory
ORDER BY pledged DESC
LIMIT 1;

-- Q5: Rank the top three countries with the most successful campaigns in terms of dollars (total amount pledged), and in terms of the number of campaigns backed.

-- Top3 countries by total amount pledged
SELECT SUM(pledged) AS total_pledged, country_id FROM cleaned_campaign
GROUP BY country_id
ORDER BY total_pledged DESC
LIMIT 3;

-- Top3 countries by number of campaigns backed
SELECT COUNT(*) AS backed_campaigns, country_id FROM cleaned_campaign
WHERE backers > 0
GROUP BY country_id
ORDER BY backed_campaigns DESC
LIMIT 3;

-- Q6: Do longer, or shorter campaigns tend to raise more money? Why?
SELECT duration_days, AVG(pledged)
FROM cleaned_campaign
GROUP BY duration_days
ORDER BY duration_days DESC;


