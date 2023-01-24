# Predicting cancer incidence associated with environmental pollutants

  
 ### Why we choose this project

A 2016 study out of Hong Kong, <a href="https://aacrjournals.org/cebp/article/25/5/839/71066/Cancer-Mortality-Risks-from-Long-term-Exposure-to">Cancer Mortality Risks From Long-term Exposure to Ambient Fine Particle</a>,  showed that long-term exposure to pollution rates of PM2.5 (ambient fine particulate matter) is associated with a higher risk of cancer in various organs. In fact, the study showed that every 10 µg/m3 of increased PM2.5 exposure <a href="https://www.aacr.org/patients-caregivers/progress-against-cancer/air-pollution-associated-cancer/">increased the risk of dying from cancer by 22%</a>.


<BR>
While the cancer risk of air pollutants is considered a clear link, the link between <a href="https://www.nature.com/articles/s41370-019-0135-4">ozone exposure and cancer incidence isn't as straightforward</a>. Studies show that while ozone increases the risk of lung cancer, exposure does not increase the risk or mortality of other cancers. 
<BR>
  <BR>
Our team's interest and experience in community health, and the environmental concerns that affect it, are what led us to create this project. We have scaled the data down to represent select counties in the United States.
<BR>
<HR>


<a name="backup">  

###  
 <Strong> Our mission is to predict the following:</strong> <br>
• Do the levels of air particulate matter affect the number of cancer instances in the United States?
<BR>
• Is the rate of cancer incidence in counties with more pollution higher than in those with moderate and lower PM rates?
<BR>
• Are specific years more impactful than others?
<BR>
• Is there a significant correlation between cancer incidence and ozone exposure?
<BR><BR>
  <strong> Team members and their roles:</strong><Br>
<a href="https://github.com/cbeckler">Charlie Beckler</a> - Team lead, engineer
<BR>
   <a href="https://github.com/meggrooms">Meg Grooms</a> - Database management, communications
<BR>
 <a href="https://github.com/katiarp">Katia Paredes</a> - ETL, modeling, web development
<BR>
 <a href="https://github.com/cmwardcode"> Cassidy Ward</a> - Modeling, web development

<BR>
 
 ## Table of Contents
1. Machine Learning Model<BR>
2. Technologies<br>
3. Communications<br>
4. Entity Relationship Diagram<BR>
5. Visualizations<BR>
6. Dashboard<BR>
7. Results<BR>
8. Challenges<BR>
9. Strengths
10. Conclusion<BR>
11. Slideshow
  
  
<hr>
  
  ### Machine Learning Model 
The goal of the Machine Learning Model is to ingest the inputs of all counties IDs/FIPS along with the max, min, and mean values for both air pollutants (PM) and Ozone (oz) and the cancer incidence trends associated for years 2001-2014. The model will then predict and output future trends of cancer incidence for each location. The selection of a machine learning model was based on several factors. The input data is labeled so a supervised learning model was selected. A classification is the best fit for our data. Support Vector Machine (SVM) was also considered because of its ability to analyze the data and label the proper class.
<BR><BR>

  
 <strong>Type of data used:</strong><BR>
 • We used <a href="#sources"> data sources</a> from Kaggle and the CDC, primarily in CSV form.
<BR><BR> 
  
  
<strong>Technologies</strong><BR>
• Python / Pandas<BR>
• PySpark <BR>
• Leaflet<BR>
• SQL
• PostgreSQL / ElephantSQL<BR>
• HTML / CSS<br>
• JavaScript<BR>
• Bootstrap<br>
• Vercel<BR>
• TeleportHQ<br>

<BR>
  
<strong>Communication Methods</strong>
<BR>
• Slack for daily communication
<BR>
• Github for sharing resources
<BR>
• Meeting 2-3 times a week via Zoom  


<BR><BR>
<HR>

  ##  <a href="https://app.quickdatabasediagrams.com/#/d/Bo92Js">Entity Relationship Diagram</a>

<a href="https://app.quickdatabasediagrams.com/#/d/Bo92Js"><img width = 600 src="https://github.com/cbeckler/eco_cancer_incidence_rates/blob/main/Database/ERD/Screenshot%202023-01-19%20at%201.59.26%20PM.png"></a>
<hr>

 ## Visualizations
 
Machine Learning Visualization<BR>  
<p align="center"
<img src="https://github.com/cbeckler/eco_cancer_incidence_rates/blob/main/Resources/tree2.png" height=800></p>
 <br><br>
  
<p align="center">
<img src="https://github.com/cbeckler/eco_cancer_incidence_rates/blob/kp_dashboard/Resources/classes_before_resampling.png"></p>
  <BR><BR>
  
<p align="center">
<img src="https://github.com/cbeckler/eco_cancer_incidence_rates/blob/kp_dashboard/Resources/classes_after_resampling1.png"></p>
<BR><BR>

<p align="center">
  <strong>Cancer Incidence Rates by County </strong>
<img src="https://github.com/cbeckler/eco_cancer_incidence_rates/blob/kp_dashboard/Resources/visualization_map.png"></p>
<BR><BR>
  
  
  
 ##  <a href="https://final-project.teleporthq.app/">Interative Dashboard</a>
  
<a href="https://final-project.teleporthq.app/"><img src="https://github.com/cbeckler/eco_cancer_incidence_rates/blob/WebDev/Webpage/Prototype/dashboard.png"></a>
  
<HR>  
<h2>Results</h2>  
  
<table>
<tr>
<td width="50%">
  
<h4>Unfortunately, the model performed poorly, with a balanced accuracy score of 0.367, only slightly better than chance (0.333)
In general, it tended to overpredict falling and rising labels, while also misclassifying some stable labels.
<Br><BR>
  
Though the model itself was not reliable, within the model it is clear ozone was more impactful than PM 2.5
The year 2006 also appeared to be the most impactful year, followed by 2009.
The chart represents years 7 and 9 of the 15 year time slice.</h4>
  </td>
  
<td width="50%">  
<img src="https://github.com/cbeckler/eco_cancer_incidence_rates/blob/main/Resources/Feature_Importances_top10.png">
 </td>
<tr>
  </table>


  
  
<hr>
  
<h2>Challenges</h2>  
Using real data rather than a curated dataset always carries the risk of finding that the variables of interest do not have an actual relationship to the outcome label. Unfortunately, that seemed to be the case.<BR><BR>
  
This was compounded by an error in between the ETL and model development stages that caused the duplication of the dataset. The error resulted in a model that initially performed well, but was not caught and fixed until late in the project development process. <BR><BR>
  
An additional error was found with how the cancer and environmental datasets were linked. We were able to fix the problem, but this did not significantly improve model performance. <BR><BR>
  
Despite the challenges we faced, the final diagnosis of poor model performance is simply that the variables do not show a relationship. Alternate models and methods were tested with little to no improvement in results.

  
<hr>
  
<h2>Strengths</h2>
Despite the disappointment and curveball of realizing the model was weak very late in the project, the team pulled together to present a flawed but interesting model.<BR><BR>
  
The team took real risks with untested data which were difficult to work with. Multiple separate, several million row, datasets were aggregated and combined to form one cohesive model dataset.<BR><BR>
  
Multiple models and methods were tested in a rigorous attempt to get a strong model.
The team used advanced visualization techniques to create <a href="https://final-project.teleporthq.app/">our own webpage</a> as a dashboard for the project with an interactive map with county data

  
<hr>
  
<h2>Conclusion</h2>  
Although it was disappointing to not get a working model, the team put together an interesting project with strong engineering behind it. We also gained invaluable expertise of how to handle projects when major issues pop up and the data ends up being less than ideal.

  
  
<hr>
  
  
<h2><a href="https://docs.google.com/presentation/d/1th906EIn5OX-S-B3GdT8gL4d71oXpQRHXfPzZ-eeQrI/edit?usp=sharing">View the Slideshow</a></h2>
  

<BR>

  
  
<a href="https://docs.google.com/presentation/d/1th906EIn5OX-S-B3GdT8gL4d71oXpQRHXfPzZ-eeQrI/edit?usp=sharing"><img src="https://github.com/cbeckler/eco_cancer_incidence_rates/blob/main/Slideshow/slideshow_title.png"></a>
<HR>


<a name="sources">
 
<BR>
<HR>
 <a href="https://data.cdc.gov/Environmental-Health-Toxicology/Daily-County-Level-Ozone-Concentrations-2001-2016/kmf5-t9yc">County-Level Ozone, All Years</a>
<BR>
<a href="https://https://www.kaggle.com/datasets/thedevastator/us-county-level-cancer-mortality-and-incidence-r?resource=download>Cancer Mortality & Incidence Rates</a>
<BR>
<a href="https://data.cdc.gov/Environmental-Health-Toxicology/Daily-PM2-5-Concentrations-All-County-2001-2016/7vdq-ztk9">Daily PM2.5 Concentrations, All Counties</a>
<BR>         
<a href="#backup"> Go back up</a>
                 
                 

