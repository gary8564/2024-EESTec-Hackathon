class PromptTemplates:
    def __init__(self):
        self.identifyPainPoints_template = None
        self.resolvePainPoints_template = None
        self.cesIndex_template = None
        self.sentimentAnalysis_template = None
        self.CxPi_template = None

    def create_identifyPainPoints_template(self):
        self.identifyPainPoints_template = """
                You are the customer experience designer assistant at Infineon. \
                Your task is to help identify the pain points of customer feedback. \
                Now, given the following customer feedback delimited by triple quotes, \
                please list customer pain points \
                and classify the possible type of customer pain points based on the provided text delimited by triple backticks.
                \"\"\"
                feedback subject: {title}
                feedback content: {body}
                \"\"\"
                ```
                text: {summary}
                ```
            """
        return self.identifyPainPoints_template
    
    def create_resolvePainPoints_template(self):
        self.resolvePainPoints_template = """
                Now, you have identified the pain points of the customer feedback as follows: \n{identifiedPainPoints} \
                please try to give short advice to resolve the customer pain points based on the provided text delimited by triple backticks.

                ```
                text: {summary}
                ```
            """
        return self.resolvePainPoints_template
    
    def create_cesIndex_template(self):
        self.cesIndex_template = """
                You are the customer experience designer assistant at Infineon. \
                Your task is to help analyze the Customer effort score (CES) of customer feedback. \
                Customer effort score measures how easy it is for a customer to use the service, \
                rated on a scale of 1 (very difficult) to 5 (very easy). \
                Now, given the following customer feedback delimited by triple backticks, \
                please output your rated CES score. 
                ```
                feedback subject: {title}
                feedback content: {body}
                ```
            """
        return self.cesIndex_template
    
    def create_sentimentAnalysis_template(self):
        self.sentimentAnalysis_template = """
                As a sentimental analyst, your task is to help analyze how does the customer feel based on the customer feedback. \
                Now, given the following customer feedback delimited by triple backticks, \
                would you categorize it as 'Positive', 'Negative', or 'Neutral'? 
                ```
                feedback subject: {title}
                feedback content: {body}
                ```
            """
        return self.sentimentAnalysis_template
    
    def create_CxPi_template(self):
        self.CxPi_template = """
                You are the customer experience designer assistant at Infineon. \
                Your task is to help analyze how effective the product srvice at Infineon was at meeting the customer's needs?. \
                Now, given the following customer feedback delimited by triple backticks, \
                please rate on a scale of 1 (very ineffective) to 5 (very effective). 
                ```
                feedback subject: {title}
                feedback content: {body}
                ```
            """
        return self.CxPi_template