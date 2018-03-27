from requests_html import HTMLSession

url = 'https://www.mikeash.com/pyblog/'
session = HTMLSession()
page = session.get(url)

friday_qa = [link for link in page.html.absolute_links if 'https://www.mikeash.com/pyblog/friday-qa' in link]
contents = []
for qa in friday_qa:
    content = session.get(qa)
    contents.append(content.html)
    print('{} downlaoded'.format(qa))

