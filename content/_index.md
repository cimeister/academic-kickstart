---
# Leave the homepage title empty to use the site title
title: Clara Meister
date: 2022-10-24
type: landing

sections:
  - block: about.biography
    id: about
    content:
      title: Biography
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin


  - block: experience
    content:
      title: Experience
      # Date format for experience
      #   Refer to https://wowchemy.com/docs/customization/#date-format
      date_format: Jan 2006
      # Experiences.
      #   Add/remove as many `experience` items below as you like.
      #   Required fields are `title`, `company`, and `date_start`.
      #   Leave `date_end` empty if it's your current employer.
      #   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
      items:
        - title: Postdoc
          company: ETH Zürich
          company_url: 
          location: Zürich, Switzerland
          date_start: 2020-09-01
          date_end: 
          description: Postdoc designing an AI project course for the AI and Data Technologies MAS degree.

        - title: PhD Student
          company: ETH Zürich
          company_url: 
          location: Zürich, Switzerland
          date_start: 2020-02-01
          date_end: 2024-11-05
          description: PhD student in the Computer Science Department (Machine Learning Institute). Helped to design and teach the [Natural Language Processing course](https://rycolab.io/classes/intro-nlp-f22/) and develop material for the [Large Language Models course](https://rycolab.io/classes/llm-s23/). Advised 13 MsC theses and various semester research projects.

        
        - title: Research Scientist Intern
          company: DeepMind
          company_url: 
          location: London, UK
          date_start: 2022-03-01
          date_end: 2022-08-01
          description: Intern with the Language Team, working with Adhi Kuncoro, Wojciech Stokowiec, and Laura Rimell.

        - title: Research Assistant
          company: ETH Zürich
          company_url: 
          location: Zürich, Switzerland
          date_start: 2019-01-01
          date_end: 2020-02-01
          description:  Research assistant in the Advanced Software Technologies Lab under Professor Zhendong Su. Area of focus was on building systems for automatically testing machine translation systems.
              
        - title: Research Assistant
          company: SLAC National Accelerator Laboratory
          company_url: 
          location: Stanford, California
          date_start: 2018-01-01
          date_end: 2018-06-30
          description: Research assistant with the Linac Coherent Light Source lab working with scientists to modify and enhance code base for analysis of electron pulse x-ray images generated at SLAC’s new hard x-ray free-electron laser.

        - title: Course Assistant
          company: Stanford University Computer Science Department
          company_url: 
          location: Stanford, California
          date_start: 2017-09-01
          date_end: 2017-12-31
          description: Course assistant for CS102 Big Data- Tools and Techniques, Discoveries and Pitfalls. Taught students basic data wrangling and analysis along with visualization techniques using multiple software platforms.
          
        - title: Data Science Intern
          company: Akamai Technologies
          location: Mountain View, California
          date_start: 2016-09-01
          date_end: 2017-06-01
          description: Part-time internship during academic school year. Projects included integrating AWS ElastiCache into infrastructure of the Data Science Team’s services and refactoring libraries to support multiple databases.

        - title: Software Engineering Intern
          company: Cisco Systems; Tendril Networks; SAP Hybris
          location: San Jose, California; Boulder, Colorado
          date_start: 2015-06-01
          date_end: 2017-09-01
          description: Summer software engineering internships 2015-2017

    design:
      columns: '2'

  - block: collection
    id: blog
    content:
      title: Blog Posts
      filters:
        folders:
          - blog
      
    design:
      columns: '2'
 
  - block: collection
    id: publications
    content:
      title: Publications
      text:  __I do not keep this updated; see my [Google Scholar](https://scholar.google.com/citations?user=quJhNH8AAAAJ&hl=en) page for an up-to-date list of publications.__
      filters:
        folders:
          - publication
        exclude_featured: false
    design:
      columns: '2'
      view: compact
  - block: markdown
    id: talks
    content:
      title: Recent & Upcoming Talks
      text: |-
        - **18/5/2023:** Invited talk at Tel Aviv University's NLP Seminar

        - **21/4/2023:** Invited talk at University of Edinburgh ILCC Seminar

        - **26/9/2022:** Invited lecture at Johns Hopkins University Artificial Agents course
        
        - **6/5/2022:** Invited talk at Google Research Machine Translation Team Reading Group

        - **14/3/2022:** Invited talk at IST-Unbabel Seminar

        - **10/8/2021:** Invited talk at UT Austin's NLP Seminar

        - **14/7/2021:** Invited talk at Berkeley's NLP Seminar

        - **16/6/2021:** Invited talk at MIT's Computational Psycholinguistics Lab
    design:
      columns: '2'
 
  
---
