import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the base URL for the first page
base_url = 'https://community.infineon.com/t5/IGBT/bd-p/IGBT'

parts = base_url.split('/')
name = parts[-1]


def web2csv(base_url):
    data = []
    page = 0

    while True:
        if page == 0:
            url = base_url
            print(url)
        else:
            # Construct the URL for the current page
            url = f'{base_url}/page/{page + 1}'
            print(url)

        response = requests.get(url, allow_redirects=False)
        # if page is not found, then break
        if response.status_code != 200:
            break

        # Request HTML content
        html_text = response.text
        soup = BeautifulSoup(html_text, 'lxml')

        # Find all product types on the current page
        product_types = soup.find_all('a', class_='board-link')

        for product_type in product_types:
            # Find the title, body, and time elements within the specific product type
            title = product_type.find_next('div', class_='subject').text.strip()
            body = product_type.find_next('div', class_='full-body body')

            if body == None:
                body = product_type.find_next('div', class_='truncated-body body').text.strip()
            else:
                body = body.text.strip()

            # time = product_type.find_next('span', class_='time').text.strip().replace(' ', '')

            data3.append({
                'Product_types': product_type.text.strip(),
                'Title': title,
                'Body': body
            })

        page += 1

        if page > 20:  # set page 20
            break

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv(f"data/{name}.csv")

    return