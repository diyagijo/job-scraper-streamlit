import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
from bs4 import BeautifulSoup
import schedule
import time
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import streamlit as st
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 1: Scrape job listings from the specified URL
def scrape_job_listings(url="https://www.karkidi.com"):
    """
    Scrape job listings from karkidi.com using selenium for JavaScript-rendered content.
    Falls back to mock data if scraping fails or no jobs are found.
    """
    jobs = []
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        driver = webdriver.Chrome(service=webdriver.chrome.service.Service(ChromeDriverManager().install()), options=chrome_options)
        driver.get(url)
        time.sleep(3)  # Wait for JavaScript to load
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()
        
        # Hypothetical selectors (replace with karkidi.com's actual HTML structure)
        job_elements = soup.find_all('div', class_='job-listing')  # Adjust class name
        for job in job_elements:
            title_tag = job.find('h2') or job.find('h3')
            company_tag = job.find('span', class_='company')
            skills_tag = job.find('div', class_='skills')
            if title_tag and company_tag and skills_tag:
                jobs.append({
                    "title": title_tag.text.strip(),
                    "company": company_tag.text.strip(),
                    "skills": skills_tag.text.strip()
                })
        
        if jobs:
            logger.info(f"Scraped {len(jobs)} jobs using selenium")
            return pd.DataFrame(jobs)
        
        logger.warning("No jobs found with selenium, falling back to mock data")
    
    except Exception as e:
        logger.error(f"Error scraping with selenium: {e}")
    
    # Fallback to mock data
    logger.info("Using mock data as fallback")
    mock_data = [
        {"title": "Software Engineer", "company": "TechCorp", "skills": "Python, Java, SQL"},
        {"title": "Data Scientist", "company": "DataInc", "skills": "Python, R, Machine Learning"},
        {"title": "Web Developer", "company": "WebWorks", "skills": "JavaScript, HTML, CSS"},
    ]
    return pd.DataFrame(mock_data)

# Step 2: Extract job details and prepare dataset
def extract_job_details():
    df = scrape_job_listings()
    df['skills'] = df['skills'].str.lower()  # Normalize skills
    return df

# Step 3: Use unsupervised clustering on skills to categorize jobs
def cluster_jobs(df, num_clusters=3):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['skills'])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['category'] = kmeans.fit_predict(X)
    joblib.dump(vectorizer, 'skills_vectorizer.pkl')
    joblib.dump(kmeans, 'kmeans_model.pkl')
    return df

# Step 4: Classify new jobs using the saved model
def classify_new_job(new_job):
    try:
        vectorizer = joblib.load('skills_vectorizer.pkl')
        kmeans = joblib.load('kmeans_model.pkl')
        new_skills = new_job['skills'].lower()
        X_new = vectorizer.transform([new_skills])
        category = kmeans.predict(X_new)[0]
        return category
    except FileNotFoundError:
        logger.error("Model or vectorizer not found, please run initial clustering first")
        return None

# Step 5: Automatically scrape new jobs daily
def daily_scrape(user_skill_preference):
    new_jobs = scrape_job_listings()
    alert_users(new_jobs, user_skill_preference)
    return new_jobs

# Step 6: Alert users if new jobs match their preferred category
def alert_users(new_jobs, user_skill_preference, placeholder=None):
    df = extract_job_details()
    df = cluster_jobs(df)
    alerts = []
    for _, new_job in new_jobs.iterrows():
        category = classify_new_job(new_job)
        if category is None:
            continue
        user_preferred_categories = df[df['skills'].str.contains(user_skill_preference, case=False)]['category'].unique()
        if category in user_preferred_categories:
            alert = f"New job '{new_job['title']}' at {new_job['company']} matches your skills ({new_job['skills']}) in category {category}"
            alerts.append(alert)
            if placeholder:
                placeholder.write(f"Alert: {alert}")
    return alerts

# Streamlit app
def run_streamlit():
    st.title("Job Scraper and Clustering System")
    st.write("Monitor job postings from karkidi.com and get alerts for jobs matching your skills.")

    # Initialize session state
    if 'jobs_df' not in st.session_state:
        st.session_state.jobs_df = extract_job_details()
        st.session_state.jobs_df = cluster_jobs(st.session_state.jobs_df)
    if 'user_skill_preference' not in st.session_state:
        st.session_state.user_skill_preference = "Python"

    # Display current jobs
    st.subheader("Current Job Listings")
    st.dataframe(st.session_state.jobs_df)

    # Skill preference input
    st.subheader("Set Your Skill Preference")
    user_skill_preference = st.text_input("Enter your skill preference (e.g., Python, JavaScript):", value=st.session_state.user_skill_preference)
    st.session_state.user_skill_preference = user_skill_preference

    # Manual scrape button
    if st.button("Scrape New Jobs Now"):
        with st.spinner("Scraping jobs from karkidi.com..."):
            new_jobs = scrape_job_listings()
            st.session_state.jobs_df = new_jobs
            st.session_state.jobs_df = cluster_jobs(st.session_state.jobs_df)
            st.success("Scraping complete!")
            st.dataframe(st.session_state.jobs_df)

    # Display alerts
    st.subheader("Job Alerts")
    placeholder = st.empty()
    alerts = alert_users(st.session_state.jobs_df, user_skill_preference, placeholder)
    if not alerts:
        placeholder.write("No new jobs match your skills yet.")

# Background scheduler for daily scraping
def run_scheduler():
    schedule.every().day.at("13:03").do(daily_scrape, user_skill_preference=st.session_state.user_skill_preference)
    while True:
        schedule.run_pending()
        time.sleep(60)

# Main execution
if __name__ == "__main__":
    # Start scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Run Streamlit app
    run_streamlit()