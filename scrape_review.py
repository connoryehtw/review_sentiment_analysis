from playwright.sync_api import sync_playwright
import datetime
import pandas as pd
import time

def scrape_google_reviews(url):
    with sync_playwright() as playwright:
        firefox = playwright.firefox
        browser = firefox.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url)

        # scrape general information of a store
        # can store it for database
        storeName_locator = 'xpath=//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div/div[1]/div[1]/h1'
        storeName = page.locator(storeName_locator).text_content().lower().replace(' ', '_')
        
        address_locator = "css=[data-item-id='address']"
        address = page.locator(address_locator).text_content()

        avgRating_locator = 'xpath=//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div/div[1]/div[2]/div/div[1]/div[2]/span[1]/span[1]'
        avgRating = float(page.locator(avgRating_locator).text_content())

        totalReview_locator = 'xpath=//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[1]/div[1]/div[2]/div/div[1]/div[2]/span[2]/span[1]/span'
        totalReview = page.locator(totalReview_locator).text_content().replace(",", "").replace("(", "").replace(")", "")
        totalReview = int(totalReview)

        scrape_dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


        # navigate to the review page
        reviewButton_locator = '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]/div/div/button[2]/div[2]/div[2]'
        page.locator(reviewButton_locator).click()
        time.sleep(10)

        name_list = []
        stars_list = []
        time_list = []
        review_list = []

        # scroll down to the bottom
        for i in range(round(totalReview / 10 - 1)+2):
            page.eval_on_selector('//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]', 'el => { el.scrollTop = el.scrollHeight; }')
            time.sleep(1)
        review_boxes = page.locator('div.jJc9Ad ')
        rb_count = review_boxes.count()
        buttons = page.query_selector_all('button:has-text("More")')

        # collect review data
        for button in buttons:
            buttons_text = button.text_content()
            # click "More" botton to expand review content
            if buttons_text.strip() == "More":
                try:
                    button.click()
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Failed to click 'More' button: {e}")
        for i in range(rb_count):
            current_review = review_boxes.nth(i)
            name = current_review.locator(".d4r55").inner_text() if current_review.locator(".d4r55") else ""
            stars = current_review.locator(".kvMYJc").get_attribute("aria-label")
            time_ = current_review.locator(".rsqaWe").inner_text()
            try:
                review = current_review.locator(".MyEned").inner_text(timeout=100)
            except:
                review = ""
                

            name_list.append(name)
            stars_list.append(stars)
            time_list.append(time_)
            review_list.append(review)

        browser.close()

        # save reviews to a dataframe
        review_df = pd.DataFrame({
            'name': name_list,
            'rating': stars_list,
            'time': time_list,
            'review': review_list
        })
        return review_df

# replace with the Google Maps URL of the business
url = "https://www.google.com.tw/maps/place/Kel's+Patriot+Pizza/@37.7316709,-83.5509437,16z/data=!4m6!3m5!1s0x88437f39a01f2363:0xb4a15aa92882b00d!8m2!3d37.7316709!4d-83.548512!16s%2Fg%2F11fnxpv3vk?entry=ttu"

review_df = scrape_google_reviews(url)
# save reviews to a csv
review_df.to_csv("review.csv", index=False)
print("finished successfully")


# --------------------------------------------------------------------------------------
# visualize distribution of rating with pie chart
import matplotlib.pyplot as plt
import pandas as pd
review_df = pd.read_csv("review.csv")
print("shape of the dataframe: ")
print(review_df.shape)
print("columns the dataframe: ")
print(review_df.columns)

rating_counts = review_df['rating'].value_counts()
plt.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=90, pctdistance = 0.7, colors=['darkslategrey', 'darkcyan', \
    'steelblue', 'lightskyblue', 'lightcyan'], shadow=True)
plt.legend(title="Ratings")
plt.title('Distribution of Ratings')
plt.show()