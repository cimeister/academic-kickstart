---
# Leave the homepage title empty to use the site title
title: ''
date: 2022-10-24
type: landing

design:
  # Default section spacing
  spacing: '6rem'

sections:
  - block: resume-biography-3
    content:
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin
      text: ''
      # Show a call-to-action button under your biography? (optional)
      headings: 
        about: About Me
    design:
      # Apply a gradient background
      css_class: hbx-bg-gradient
      # Avatar customization
      avatar:
        size: medium # Options: small (150px), medium (200px, default), large (320px), xl (400px), xxl (500px)
        shape: circle # Options: circle (default), square, rounded

  - block: collection
    id: blog
    content:
      title: Blog Posts
      filters:
        folders:
          - blog
      
    design:
      columns: '2'
      view: list
 
  # - block: collection
  #   id: publications
  #   content:
  #     title: Publications
  #     text:  __I do not keep this updated; see my [Google Scholar](https://scholar.google.com/citations?user=quJhNH8AAAAJ&hl=en) page for an up-to-date list of publications.__
  #     filters:
  #       folders:
  #         - publications
  #       exclude_featured: false
  #   design:
  #     columns: '2'
  #     view: citation
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
