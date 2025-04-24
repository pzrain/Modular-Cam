import openai
import os

SECRET_KEY = ''
def send_to_chat(
    model: str, messages: list, max_tokens: int = 2048, temperature: float = 0.5
) -> str:
    openai.api_key = "EMPTY"
    openai.api_base = "http://localhost:8000/v1"

    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    response_content = response.choices[0].message.content
    d = response['usage'].to_dict()
    d['content'] = response_content
    return d

def send_to_chat_gpt(
    model: str, messages: str, max_tokens: int = 2048, temperature: float = 0.5
) -> str:
    openai.api_key = SECRET_KEY
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    response_content = response.choices[0].message.content
    d = response['usage'].to_dict()
    d['content'] = response_content
    return d

def send_to_chat_gpt_intruct(
    model: str, messages: str, max_tokens: int = 2048, temperature: float = 0.5
) -> str:
    openai.api_key = SECRET_KEY
    # response = openai.ChatCompletion.create(
    #     model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    # )
    response = openai.Completion.create(
        model=model,
        prompt=messages[0]['content'],
        temperature=temperature,
        max_tokens=max_tokens,
    ) 
    response_content = response.choices[0].text
    d = response['usage'].to_dict()
    d['content'] = response_content
    return d

def send_prompt(model, prompt, *args, **kwargs):
    messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    if 'gpt' in model:
        if 'instruct' in model:
            response = send_to_chat_gpt_intruct(model,messages, *args, **kwargs)
        else:
            response = send_to_chat_gpt(model,messages, *args, **kwargs)
    else:
        response = send_to_chat(model,messages, *args, **kwargs)
    return response

if __name__ == '__main__':
    prompt = '''Task Description: Rearrange a provided list of items according to the user's preferences, making minimal changes to the original order. The user's history (ordered from oldest to latest) and descriptions of each interacted item will be given.
Input Format:Item Descriptions List: A list of tuples with each tuple containing an item ID and its description, in the format [(item id, item description), ...]. For example, [(1, "Compact digital camera"), (2, "Wireless ergonomic keyboard"), ...]; User History List: A list of strings representing the descriptions of user's watched/purchased/viewed items, ordered from the oldest to the most recent. For example, ["Wireless ergonomic keyboard", "violin", ...].
Instructions for GPT: Analyze the user's history, noting any patterns or preferences. Based on this analysis, reorder the item descriptions list so that items aligning most closely with the user's demonstrated preferences are prioritized, but make only necessary changes to preserve the original order as much as possible. Provide the output as a list of item IDs, reordered according to the user's preferences. You should contain all the item IDs that have appeared in the item descriptions list. You should bear in mind that you may not change the original item descriptions list if you are not so sure.
Expected Output: A list of item IDs in the order that best matches the user's preferences. For example, [1, 2, 0, 3].
Example Input: Item Descriptions List: [(0, "Compact digital camera"), (1, "Wireless ergonomic keyboard"), (2, "Bluetooth sports headphones"), (3, "High-definition action camera")]; User History List: ["Low-definition action camera", "Pocket Camera", "Bluetooth sports headphones", "Wireless ergonomic keyboard"]
Example Task for GPT: Given the user's recent purchase and high interest in photography and sports equipment, reorder the item descriptions list to reflect these interests.
Example Expected Output:[1, 2, 0, 3]
Input: Item Descriptions List: [(1, "Pocahontas 1995::Animation|Children's|Musical|Romance"),(25, "James and the Giant Peach 1996::Animation|Children's|Musical"),(3, "Return of Jafar  The 1993::Animation|Children's|Musical"),(39, "Wild America 1997::Adventure|Children's"),(67, "Road to El Dorado  The 2000::Animation|Children's"),(76, "Evita 1996::Drama|Musical"),(53, "Bad Moon 1996::Horror"),(64, "Dangerous Minds 1995::Drama"),(84, "EDtv 1999::Comedy"),(85, "GoodFellas 1990::Crime|Drama"),(74, "Benny & Joon 1993::Comedy|Romance"),(43, "Red Corner 1997::Crime|Thriller"),(5, "Midsummer Night's Dream  A 1999::Comedy|Fantasy"),(75, "Anaconda 1997::Action|Adventure|Thriller"),(77, "Whole Wide World  The 1996::Drama"),(70, "Gordy 1995::Comedy"),(2, "Bulworth 1998::Comedy"),(33, "Three Colors: Red 1994::Drama"),(29, "Double Happiness 1994::Drama"),(27, "Phenomenon 1996::Drama|Romance")]; User History List: ["Titanic 1997::Drama|Romance","Cinderella 1950::Animation|Children's|Musical","Meet Joe Black 1998::Romance","Last Days of Disco  The 1998::Drama","Erin Brockovich 2000::Drama","Christmas Story  A 1983::Comedy|Drama","To Kill a Mockingbird 1962::Drama","One Flew Over the Cuckoo's Nest 1975::Drama","Wallace & Gromit: The Best of Aardman Animation 1996::Animation","Star Wars: Episode IV - A New Hope 1977::Action|Adventure|Fantasy|Sci-Fi","Wizard of Oz  The 1939::Adventure|Children's|Drama|Musical","Fargo 1996::Crime|Drama|Thriller","Run Lola Run Lola rennt 1998::Action|Crime|Romance","Rain Man 1988::Drama","Saving Private Ryan 1998::Action|Drama|War","Awakenings 1990::Drama","Gigi 1958::Musical","Sound of Music  The 1965::Musical","Driving Miss Daisy 1989::Drama","Bambi 1942::Animation|Children's","Apollo 13 1995::Drama","Mary Poppins 1964::Children's|Comedy|Musical","E.T. the Extra-Terrestrial 1982::Children's|Drama|Fantasy|Sci-Fi","My Fair Lady 1964::Musical|Romance","Ben-Hur 1959::Action|Adventure|Drama","Big 1988::Comedy|Fantasy","Sixth Sense  The 1999::Thriller","Dead Poets Society 1989::Drama","James and the Giant Peach 1996::Animation|Children's|Musical","Ferris Bueller's Day Off 1986::Comedy","Secret Garden  The 1993::Children's|Drama","Toy Story 2 1999::Animation|Children's|Comedy","Airplane! 1980::Comedy","Pleasantville 1998::Comedy","Dumbo 1941::Animation|Children's|Musical","Princess Bride  The 1987::Action|Adventure|Comedy|Romance","Snow White and the Seven Dwarfs 1937::Animation|Children's|Musical","Miracle on 34th Street 1947::Drama","Ponette 1996::Drama","Schindler's List 1993::Drama|War","Beauty and the Beast 1991::Animation|Children's|Musical","Tarzan 1999::Animation|Children's","Close Shave  A 1995::Animation|Comedy|Thriller","Aladdin 1992::Animation|Children's|Comedy|Musical","Toy Story 1995::Animation|Children's|Comedy","Bug's Life  A 1998::Animation|Children's|Comedy","Antz 1998::Animation|Children's","Hunchback of Notre Dame  The 1996::Animation|Children's|Musical","Hercules 1997::Adventure|Animation|Children's|Comedy|Musical","Mulan 1998::Animation|Children's"].
Task: Given the titles and genres of user's recently watched movies in the user history list, reorder the item descriptions list to reflect these interests, but keep changes to a minimum.
Output:'''
    model = "gpt-3.5-turbo-1106"
    res = send_prompt(model, prompt, temperature = 0.7)
    print(res['content'])