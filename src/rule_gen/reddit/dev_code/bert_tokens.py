from transformers import BertTokenizer

def count_bert_tokens(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return len(tokens)


def main():
    text1 = """
    <text1>1 Submissions must be about technology	Submissions must be primarily news and developments relating to technology. Self posts must contribute positively to r/technology and foster reasonable discussion. Cross-post self posts are not acceptable. Submissions relating to business and politics must be sufficiently within the context of technology in that they either view the events from a technological standpoint or analyse the  repercussions in thetechnological world.
2 No images, audio, or video	Articles with supporting image and video content are allowed; if the text is only there to explain the media, then it is not suitable. A good rule of thumb is to look at the URL; if it's a video hosting site, or mentions video in the URL, it's not suitable.
3 Titles must be taken directly from the article	Submissions must use either the articles title and optionally a subtitle, or, only if neither are accurate, a suitable quote,  which must: adequately describe the content adequately describe the content's relation to technology be free of user editorialization or alteration of meaning.
4 No technical support or help questions	Requests for tech support, asking questions or asking for help: submit to r/techsupport , r/AskTechnology , another relevant community or our weekly Support Saturday threads.
5 No petitions, surveys, or crowdfunding	Please do not submit any petitions, surveys, crowdfunding, or any other call to action or fundraising.
6 No customer support or feedback	Please do not submit discussions of one or more incidents of customer support or customer feedback.
7 No directed abusive language	You are advised to abide by reddiquette ; it will be enforced when user behavior is no longer deemed to be suitable for a technology forum. Remember; personal attacks, directed abusive language, trolling or bigotry in any form, are therefore not allowed and will be removed.

"""
    text2 = """
    Submissions must focus on technology news or developments. Business or political topics are allowed only if analyzed from a technological perspective. Cross-posted self posts aren't permitted.
Articles with supporting media are allowed, but media-only submissions (e.g., videos) are not.
Titles must be taken directly from the article or a suitable quote that accurately reflects its content and tech relevance, without edits.
No tech support questions—submit those to r/techsupport, r/AskTechnology, or Support Saturday threads.
No petitions, surveys, crowdfunding, or calls to action.
No customer support or feedback discussions.
Follow reddiquette; no personal attacks, abusive language, trolling, or bigotry.
"""
    text3 = """
    Submissions must focus on technology news or developments. Business or political topics are allowed only if analyzed from a technological perspective. Cross-posted self posts are not permitted and must contribute positively to the community by fostering reasonable discussion.

Articles with supporting media are allowed, but submissions that primarily consist of media (e.g., videos) without substantial text are not suitable. If the text only explains the media, the submission is not allowed. A good rule of thumb: if the URL is from a video hosting site or mentions video, it’s not acceptable.

Titles must be taken directly from the article, either the title or the subtitle. If neither accurately describes the content, a suitable quote may be used, provided it adequately reflects the content, its relevance to technology, and is free of any editorialization or alteration of meaning.

No tech support questions—please submit these to r/techsupport, r/AskTechnology, or other relevant communities, including our weekly Support Saturday threads.

No petitions, surveys, crowdfunding, or any other calls to action.

No discussions involving customer support incidents or feedback about specific companies.

Follow reddiquette; it will be enforced when user behavior is no longer deemed suitable for a technology forum. Personal attacks, directed abusive language, trolling, or bigotry in any form are strictly prohibited and will be removed.

"""
    for t in [text1,text2,text3]:
        print(count_bert_tokens(t))


if __name__ == "__main__":
    main()