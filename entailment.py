import requests
import json

def get_list(sen):
	"""
	Converts the sentence into a List such that it can be used as an POST request json blob data.

	Args:
		sen : a string

	Returns:
		res: a list of list containg words and tags.
	"""
	res = list()
	count = 0
	for word in sen.split():
	    l = [word, "Any", count]
	    count += 1
	    res.append(l)
	return res

def get_ai2_textual_entailment(t, h):
	"""
	Returns the output of POST request to AI2 textual entailment service

	Args:
		t, h : text and hypothesis (two strings)

	Returns:
		req : A text version of json response.
	"""
	text = getList(t)
	hypothesis = getList(h)

	data = { "text" : text, "hypothesis": hypothesis}

	headers = {'Content-type': 'application/json', 'Accept': 'application/json'}

	url = 'http://entailment.prod.allenai.org:8191/api/entails'

	req = requests.post(url, headers=headers, data=json.dumps(data))

	return req.json()

def main():

	text = raw_input("Enter the text: ")
	hypothesis = raw_input("Enter the hypothesis: ")

	print "Response: "

	print getTextualEntailment(text, hypothesis)


if __name__ == "__main__":
	main()
