# test_vader_sentiment.py

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import matplotlib.pyplot as plt
    import numpy as np

    def check_vader_sentiment(reviews):
        # Initialize the SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()

        # Lists to store aggregated sentiment scores
        neg_scores = []
        neu_scores = []
        pos_scores = []
        compound_scores = []

        # Perform sentiment analysis on each review
        for review in reviews:
            sentiment_score = analyzer.polarity_scores(review)
            print(f"Sentiment analysis result for review: {review}")
            print(sentiment_score)
            neg_scores.append(sentiment_score['neg'])
            neu_scores.append(sentiment_score['neu'])
            pos_scores.append(sentiment_score['pos'])
            compound_scores.append(sentiment_score['compound'])

        # Calculate overall average sentiment scores
        avg_neg = np.mean(neg_scores)
        avg_neu = np.mean(neu_scores)
        avg_pos = np.mean(pos_scores)
        avg_compound = np.mean(compound_scores)

        print("\nOverall average sentiment scores:")
        print(f"Negative: {avg_neg:.3f}")
        print(f"Neutral: {avg_neu:.3f}")
        print(f"Positive: {avg_pos:.3f}")
        print(f"Compound: {avg_compound:.3f}")

        # Plot the results for each review
        index = np.arange(len(reviews))
        bar_width = 0.2

        plt.bar(index, neg_scores, bar_width, color='red', label='Negative')
        plt.bar(index + bar_width, neu_scores, bar_width, color='gray', label='Neutral')
        plt.bar(index + 2 * bar_width, pos_scores, bar_width, color='green', label='Positive')
        plt.bar(index + 3 * bar_width, compound_scores, bar_width, color='blue', label='Compound')

        plt.xlabel('Reviews')
        plt.ylabel('Scores')
        plt.title('Sentiment Analysis Results')
        plt.xticks(index + bar_width, [f'Review {i+1}' for i in range(len(reviews))])
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Plot the overall average sentiment scores
        categories = ['Negative', 'Neutral', 'Positive', 'Compound']
        averages = [avg_neg, avg_neu, avg_pos, avg_compound]

        plt.bar(categories, averages, color=['red', 'gray', 'green', 'blue'])
        plt.xlabel('Sentiment Categories')
        plt.ylabel('Average Scores')
        plt.title('Overall Average Sentiment Scores')
        plt.show()

    if __name__ == "__main__":
        # Sample list of reviews
        reviews = [
            "The movie was great! I really enjoyed it.",
            "It was an average movie, not too bad.",
            "I did not like the movie at all. It was boring.",
            "The plot was interesting, but the acting was terrible."
        ]
        check_vader_sentiment(reviews)
        print("VADER Sentiment module is working correctly.")
except ImportError:
    print("vaderSentiment module is not installed.")
