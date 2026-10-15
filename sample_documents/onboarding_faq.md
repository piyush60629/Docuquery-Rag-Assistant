# Data Engineering Onboarding FAQ

Answers to common questions from engineers joining the retail analytics team. All details are synthetic.

## How do I get access to the platform?

Raise an access request through the service portal and select the Retail Analytics Engineer role. Your manager approves the request, and access to the development environment is usually ready within two working days. Production access is read-only and requires a separate approval from the platform owner.

## Which environments exist?

There are three environments: development, QA, and production. Engineers build and test changes in development. QA runs regression tests on a copy of the last 30 days of data. Only the deployment pipeline in Azure DevOps can change production.

## What is the service level for reports?

Gold tables must be ready by 6:00 AM IST every day so that store managers can see the previous day's sales in their morning report. If the pipeline is at risk of missing this time, the on-call engineer informs the reporting team by 5:30 AM.

## How does the on-call rotation work?

Each engineer is on call for one week at a time, from Monday 9:00 AM to the next Monday 9:00 AM. The on-call engineer monitors pipeline alerts, handles failed runs, and writes a short incident summary for every failure that affects the Gold layer. New joiners shadow an experienced engineer for their first on-call week.

## What are the naming conventions?

- Pipelines start with pl_ followed by the layer and entity, for example pl_silver_sales.
- Databricks notebooks start with nb_, for example nb_gold_sales_fact.
- Tables in the Gold layer use the prefixes fact_ and dim_.

## How do I work with Git?

Create a feature branch from main using the format feature/ticket-number-short-description. Every change needs a pull request with at least one approval before it is merged. Direct commits to main are blocked.

## Where do I ask questions?

Use the team channel for general questions and the data quality channel for data issues. For access problems, contact the platform owner.
