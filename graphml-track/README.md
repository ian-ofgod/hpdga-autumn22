# Graph ML Challenge: Automatic Graph Expansion

## Description
The goal of this challenge consists in implementing a system for Automatic Graph Expansion for the detection of Financial Fraud Cases. 
Such a system can help banks to detect potential financial crimes and report them to proper authorities.

## Dataset
You will work on a Financial Graph representing bank's operations.

The graph has 319376 nodes, and 921876 edges, listed in the files:
* *polimi.case.graphs.vertices.csv* 
* *polimi.case.graphs.edges.csv*

There are 5 types of nodes and 4 types of edges in the graph, identified by the `Label` column in the two files.
Each type of node and edge has specific properties, as illustrated in the following tables:

| Node Type         | Property 1        | Property 2             | Property 3       |
|-------------------|-------------------|------------------------|------------------|
| *Customer*        | Name              | Person or Organisation | Income Size Flag |
| *Derived Entity*  | Name              | Person or Organisation |                  |
| *External Entity* | Name              | Person or Organisation |                  |
| *Account*         | Account ID String | Revenue Size Flag      |                  |
| *Address*         | Address           |                        |                  |

| Edge Type        | Property 1          |
|------------------|---------------------|
| *Money Transfer* | Amount Flag         |
| *Is Similar*     | Similarity Strength |
| *Has Account*    |                     |
| *Has Address*    |                     |

Bank's operation data are supplemented by external data about potential fraudulent cases.
Each **case** basically is a subgraph of the financial graph, including potentially problematic entities and relations among them.
*Core nodes* belonging to a *core case* are marked by the same value in `CoreCaseGraphID` property.

Core case graphs should be expanded with nodes that are not flagged but are relevant to the case.
The identification of such relevant nodes (i.e.: *case expansion*) is crucial to the correct decision whether the present case has legitimate explanation or is case of financial crime.
The *extended nodes* to include in a final case graph are marked by the same value in `ExtendedCaseGraphID` property, which represents the ground truth information to use for learning.

In the financial graph provided to you, 4000 cases (i.e.: subgraphs) have been identified; then they have been split in *training cases* (3013 cases) and *testing cases* (987 cases).  
Each set is a represented by a pair of csv files with `NodeID <--> CaseID` mapping, one file for core nodes, and the other one for extended nodes.
- For the training cases, there are `training.core.vertices.csv` and `training.extended.vertices.csv` files, and you can see that info also in the graph (`CoreCaseGraphID` and `ExtendedCaseGraphID` properties). Nodes included in the training cases have the `testingFlag` property set to 0. You can use this nodes for the training and evaluation of your models.
- For the testing cases, we provide you the `testing.core.vertices.csv` file for core cases, and you can see that info also in the graph (`CoreCaseGraphID` property). Nodes included in the testing cases have the `testingFlag` property set to 1. Your models must predict which nodes of the graph should expand each testing core case. The `testing.extended.vertices.csv` file, containing all the correct predictions, will be used for final evaluation and will be made available after the end of the contest.

**NOTE:** We have noticed little inconsistencies between the `polimi.case.graphs.vertices.csv` data and the `training.core.vertices.csv`, `training.extended.vertices.csv`, and `testing.core.vertices.csv` data. This is due to the fact that a node can be a core or extended node in multiple case subgraphs,both in training or testing cases. However, the `CoreCaseGraphID`, `ExtendedCaseGraphID`, and `testingFlag` properties store just one value. Thus, if a node is a core node of two cases (e.g. cases 420 and 1995), in the `CoreCaseGraphID` you will see only the case with the highest ID value (1995 in this case). The same happens for `ExtendedCaseGraphID`, while in the `testingFlag` feature you will only see the value 1 even if the nodes is part both of the training and the testing set (for different case subgraphs). To avoid any problem, we suggest to delete all the info included in the `CoreCaseGraphID`, `ExtendedCaseGraphID`, and `testingFlag` properties. Then you can import the correct information for the training cases from the `training.core.vertices.csv` and `training.extended.vertices.csv` files, and for the testing cases from the `testing.core.vertices.csv`.

## Challenge
The goal of this challenge consists in implementing a system for automatic graph expansion that, given a graph and a set of subgraphs, automatically finds the nodes of the graph that should expand each subgraph.
In detail, for each testing core case subgraph, your models must predict which nodes of the Financial Graph should expand that subgraph. 

### Performance Metric
The performance of your algorithm is evaluated using the F1 score: https://en.wikipedia.org/wiki/F-score

In detail, *True Positives*, *False Positives*, and *False Negatives*, employed for computing the F1 score, are defined according to the following table:

|                           | **Predicted part of case** | **Predicted not in case** |
|---------------------------|----------------------------|---------------------------|
| **Actually part of case** | True Positive              | False Negative            |
| **Actually not in case**  | False Positive             | True Negative             |

Any other evaluation that allows deeper understanding the performance of your models is welcome!

## Submission
You must submit your solution by January 10th 2023 at 11.59 PM, while the deadline for the development is December 31st 2022 11.59 PM (no commits to code after the deadline!). 

The submission consists in an email to Guido Walter Di Donato (`guidowalter.didonato at polimi.it`), with CC to Ian Di Dio Lavore (`ian.didio at polimi.it`) and Marco Santambrogio (`marco.santambrogio at polimi.it`),
containing the name of the partecipants, and a link to the `git` repository on which you developed your solution (if private, add `gwdidonato` and `ian-ofgod` as collaborators).

The repository must contain:
* The source code
* A README that explains how to execute your solution (we will use the same data we provided to you)
* A pdf report (or a detailed readme, but we suggest the report!) describing your approach, your models, your results, and any other implementative decision you made that you'd like to share with us. Remember to justify your choices: the easier for us to understand your work, the better it is for you!
* A CSV file containing your predictions for nodes extending each testing core case
    * The structure of the CSV file is as follows

| NodeID  | ExtendedCaseGraphID |
|---------|---------------------|
| N12345  | 1                   |
| N2345   | 1                   |
| N345678 | 5                   |

Please note:
* External libraries are allowed: please report name and version of such libraries in the README. 
* Your code and tests must be runnable using a bash or python script.
* Keep in mind: the easier for us to replicate your results, the better it is for you!
* The report should be 6 pages long at most, with minimum font size 11 pt.
* The `demo_graphml.ipynb` contains the demo shown during the 14th November lecture.

