wikiartList = [
               "Name: Full name of the artist,or leave empty if not applicable", 
               "Birth_Date: Date in %Y/%-m/%-d format, e.g., 1839/1/9,or leave empty if not applicable",
               "Death_Date: Date in %Y/%-m/%-d format, e.g., 1836/1/1, or leave empty if not applicable",
               "Age: Age the age of the artist e.g.,87, or leave empty if not applicable",
               "Birth_Country: [Country where the artist was born] eg,.Netherlands,or leave empty if not applicable",
               "Death_Country: [Country where the artist died, or leave empty if not applicable] eg,. Switzerland,or leave empty if not applicable",
               "Birth_City: [City where the artist was born] ,or leave empty if not applicable",
               "Death_City: [City where the artist died, or leave empty if not applicable]",
               "Field: [Primary artistic field, e.g., Painting, Sculpture],or leave empty if not applicable",
               "Genre: [Primary genre of work, e.g., Portrait, Landscape],or leave empty if not applicable",
               "Marriage: [Marital status or details, or leave empty if not applicable,as succinct as possible]",
               "Art_Movement: [Art movement contributed to, e.g., Impressionism],or leave empty if not applicable"
               ]

nba_list = [
         "name: the name of the nba player", 
         "birth_date: the birthday of the nba player e.g., June 10, 1959",
         "nationality: the nation of the nba player.eg., American,or leave empty if not applicable",
         "age: the age of the nba player. eg., 34,this year is 2024 ",
         "team: the current team of the nba player.eg.,Lokomotiv Kuban,or leave empty if not applicable",
         "position: player’s position on the court. eg., foeword,or leave empty if not applicable",
         "draft_pick: the draft pick of the nba player eg.,24. ,or leave empty if not applicable",
         "draft_year: the the draft year of the nba player. eg.,2015,or leave empty if not applicable",
         "college: the college of the nba player. eg.,University of Notre Dame,or leave empty if not applicable",
         "NBA_championships:How many times has this NBA player won NBA championships. eg.,1,or 0 empty if not applicable",
         "mvp_awards: How many times has this NBA player won MVP award. eg.,2,or leave 0 if not applicable",
         "olympic_gold_medals: How many times has this NBA player won an Olympic gold medal. eg.,2,or leave 0 if not applicable",
         "FIBA_World_Cup: How many times has this NBA player won FIBA World Cup. eg,.1,or leave 0 if not applicable",
               ]

legal_list = ["judge_name  VARCHAR, judge_name is the name of the judge presiding over the case. e.g., arshall J.or leave empty if not applicable", 
               "plaintiff VARCHAR, plaintiff is the name of the person or organization initiating the case. e.g., [Australian Competition and Consumer Commission].or leave empty if not applicable",
               "defendant VARCHAR, defendant is the name of the person or entity being sued or accused. e.g., [Narnia Investments Pty Ltd].or leave empty if not applicable",
               "hearing_year DATE, hearing_year is the date when the hearing began (first day if multiple days).in %Y/%-m/%-d format. e.g., 2009/4/22.or leave empty if not applicable",
               "judgment_year DATE, judgment_year is the date when the judgment was delivered (first day if multiple days).in %Y/%-m/%-d format. e.g.,  2009/4/22.or leave empty if not applicable",
               "case_type ENUM('Criminal Case', 'Civil Case', 'Commercial Case', 'Administrative Case'), case_type identifies the type of the case. e.g., Criminal",
               "verdict ENUM('Guilty', 'Not Guilty', 'Others', 'Dismissed', 'Approved', 'Others'), verdict indicates the court's decision. e.g., Guilty",
               "counsel_for_applicant VARCHAR, counsel_for_applicant is the name of the applicant’s lawyer. e.g., Mr I Faulkner SC.or leave empty if not applicable",
               "counsel_for_respondent VARCHAR, counsel_for_respondent is the name of the respondent’s lawyer. e.g., Mr D Barclay.or leave empty if not applicable",
               "nationality_for_applicant VARCHAR, nationality_for_applicant is the applicant’s nationality. e.g., Australia.or leave empty if not applicable",
               "hearing_location VARCHAR, hearing_location is the location where the hearing occurred. e.g., Hobart.or leave empty if not applicable",
               "evidence ENUM('1', '0'), evidence specifies whether evidence was provided in the case.1 for yes, 0 for no. e.g., 1",
               "first_judge ENUM('1', '0'), first_judge indicates if it was the first judgment or a subsequent one. 1 for yes, 0 for no. e.g., 1"
               ]

fin_list = ['Company Name: The name of the company, add “Inc.” or “Company” as a suffix. or leave empty if not applicable',
               'Founding Year: The year the company was founded (e.g., “2003”). or leave empty if not applicable', 
               'Headquarters Location: The headquarters address, as it is written in the original text.or leave empty if not applicable', 
               'Industry Type: The industry category of the company (Categories include:Energy andResources; Industrial and Manufacturing; Real Estate and Construction; Retaiand Consumer;)Multiple categories can be selected.or leave empty if not applicable', 
               'Product/Service Type: The company’s products, which can be inferred from the opening of each paragraph.or leave empty if not applicable', 
               'Customer Type: The type of customers the company serves.or leave empty if not applicable',
               'Tax Policy: Whether the company has a special tax policy. If yes, fill 1; if no, fill 0.', 
               'Branches Office: Whether the company has branch offices. If yes, fill 1; if no, fill 0.',
               'Brand Number: The number of brands the company has, in numerical form.if no, fill 0.', 
               'Changed Name: Whether the company has changed its name. If yes, fill 1; if no, fill 0.', 
               'Segments Number: The number of departments or divisions the company has, in numerical form.if no, fill 0.']



datasets_attributes = {
"fin" :['Company Name', 
               'Founding Year', 
               'Headquarters Location',
               'Industry Type', 
               'Product/Service Type', 
               'Customer Type', 
               'Tax Policy',
               'Branches Office', 
               'Brand Number', 
               'Changed Name', 
               'Segments Number'],

"legal":["judge_name", 
            "plaintiff",
            "defendant",
            "hearing_year",
            "judgment_year",
            "case_type",
            "verdict",
            "counsel_for_applicant",
            "counsel_for_respondent",
            "nationality_for_applicant",
            "hearing_location",
            "evidence",
            "first_judge"
            ],

"nba":[
            "name", 
            "birth_date",
            "nationality",
            "age",
            "team",
            "position",
            "draft_pick",
            "draft_year",
            "college",
            "NBA_championships",
            "mvp_awards",
            "olympic_gold_medals",
            # "FIBA_World_Cup",
            ],

"wikiart":[
            "Name", 
            "Birth_Date",
            "Death_Date",
            "Age",
            "Birth_Country",
            "Death_Country",
            "Birth_City",
            "Death_City",
            "Field",
            "Genre",
            "Marriage",
            "Art_Movement"
            ]
}
