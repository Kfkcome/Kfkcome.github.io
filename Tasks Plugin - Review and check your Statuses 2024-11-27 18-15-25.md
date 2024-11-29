# Review and check your Statuses

## About this file

This file was created by the Obsidian Tasks plugin (version 7.14.0) to help visualise the task statuses in this vault.

If you change the Tasks status settings, you can get an updated report by:

- Going to `Settings` -> `Tasks`.
- Clicking on `Review and check your Statuses`.

You can delete this file any time.

## Status Settings

<!--
Switch to Live Preview or Reading Mode to see the table.
If there are any Markdown formatting characters in status names, such as '*' or '_',
Obsidian may only render the table correctly in Reading Mode.
-->

These are the status values in the Core and Custom statuses sections.

| Status Symbol | Next Status Symbol | Status Name | Status Type | Problems (if any) |
| ----- | ----- | ----- | ----- | ----- |
| `space` | `x` | Todo | `TODO` |  |
| `x` | `space` | Done | `DONE` |  |
| `/` | `x` | In Progress | `IN_PROGRESS` |  |
| `-` | `space` | Cancelled | `CANCELLED` |  |
| `space` | `x` | Unchecked | `TODO` | Duplicate symbol '`space`': this status will be ignored. |
| `x` | `space` | Checked | `DONE` | Duplicate symbol '`x`': this status will be ignored. |
| `>` | `x` | Rescheduled | `TODO` |  |
| `<` | `x` | Scheduled | `TODO` |  |
| `!` | `x` | Important | `TODO` |  |
| `?` | `x` | Question | `TODO` |  |
| `*` | `x` | Star | `TODO` |  |
| `n` | `x` | Note | `TODO` |  |
| `l` | `x` | Location | `TODO` |  |
| `i` | `x` | Information | `TODO` |  |
| `I` | `x` | Idea | `TODO` |  |
| `S` | `x` | Amount | `TODO` |  |
| `p` | `x` | Pro | `TODO` |  |
| `c` | `x` | Con | `TODO` |  |
| `b` | `x` | Bookmark | `TODO` |  |
| `"` | `x` | Quote | `TODO` |  |
| `0` | `0` | Speech bubble 0 | `NON_TASK` |  |
| `1` | `1` | Speech bubble 1 | `NON_TASK` |  |
| `2` | `2` | Speech bubble 2 | `NON_TASK` |  |
| `3` | `3` | Speech bubble 3 | `NON_TASK` |  |
| `4` | `4` | Speech bubble 4 | `NON_TASK` |  |
| `5` | `5` | Speech bubble 5 | `NON_TASK` |  |
| `6` | `6` | Speech bubble 6 | `NON_TASK` |  |
| `7` | `7` | Speech bubble 7 | `NON_TASK` |  |
| `8` | `8` | Speech bubble 8 | `NON_TASK` |  |
| `9` | `9` | Speech bubble 9 | `NON_TASK` |  |

## Loaded Settings

<!-- Switch to Live Preview or Reading Mode to see the diagram. -->

These are the settings actually used by Tasks.

```mermaid
flowchart LR

classDef TODO        stroke:#f33,stroke-width:3px;
classDef DONE        stroke:#0c0,stroke-width:3px;
classDef IN_PROGRESS stroke:#fa0,stroke-width:3px;
classDef CANCELLED   stroke:#ddd,stroke-width:3px;
classDef NON_TASK    stroke:#99e,stroke-width:3px;

1["'Todo'<br>[ ] -> [x]<br>(TODO)"]:::TODO
2["'Done'<br>[x] -> [ ]<br>(DONE)"]:::DONE
3["'In Progress'<br>[/] -> [x]<br>(IN_PROGRESS)"]:::IN_PROGRESS
4["'Cancelled'<br>[-] -> [ ]<br>(CANCELLED)"]:::CANCELLED
5["'Rescheduled'<br>[&gt;] -> [x]<br>(TODO)"]:::TODO
6["'Scheduled'<br>[&lt;] -> [x]<br>(TODO)"]:::TODO
7["'Important'<br>[!] -> [x]<br>(TODO)"]:::TODO
8["'Question'<br>[?] -> [x]<br>(TODO)"]:::TODO
9["'Star'<br>[*] -> [x]<br>(TODO)"]:::TODO
10["'Note'<br>[n] -> [x]<br>(TODO)"]:::TODO
11["'Location'<br>[l] -> [x]<br>(TODO)"]:::TODO
12["'Information'<br>[i] -> [x]<br>(TODO)"]:::TODO
13["'Idea'<br>[I] -> [x]<br>(TODO)"]:::TODO
14["'Amount'<br>[S] -> [x]<br>(TODO)"]:::TODO
15["'Pro'<br>[p] -> [x]<br>(TODO)"]:::TODO
16["'Con'<br>[c] -> [x]<br>(TODO)"]:::TODO
17["'Bookmark'<br>[b] -> [x]<br>(TODO)"]:::TODO
18["'Quote'<br>[&quot;] -> [x]<br>(TODO)"]:::TODO
19["'Speech bubble 0'<br>[0] -> [0]<br>(NON_TASK)"]:::NON_TASK
20["'Speech bubble 1'<br>[1] -> [1]<br>(NON_TASK)"]:::NON_TASK
21["'Speech bubble 2'<br>[2] -> [2]<br>(NON_TASK)"]:::NON_TASK
22["'Speech bubble 3'<br>[3] -> [3]<br>(NON_TASK)"]:::NON_TASK
23["'Speech bubble 4'<br>[4] -> [4]<br>(NON_TASK)"]:::NON_TASK
24["'Speech bubble 5'<br>[5] -> [5]<br>(NON_TASK)"]:::NON_TASK
25["'Speech bubble 6'<br>[6] -> [6]<br>(NON_TASK)"]:::NON_TASK
26["'Speech bubble 7'<br>[7] -> [7]<br>(NON_TASK)"]:::NON_TASK
27["'Speech bubble 8'<br>[8] -> [8]<br>(NON_TASK)"]:::NON_TASK
28["'Speech bubble 9'<br>[9] -> [9]<br>(NON_TASK)"]:::NON_TASK
1 --> 2
2 --> 1
3 --> 2
4 --> 1
5 --> 2
6 --> 2
7 --> 2
8 --> 2
9 --> 2
10 --> 2
11 --> 2
12 --> 2
13 --> 2
14 --> 2
15 --> 2
16 --> 2
17 --> 2
18 --> 2
19 --> 19
20 --> 20
21 --> 21
22 --> 22
23 --> 23
24 --> 24
25 --> 25
26 --> 26
27 --> 27
28 --> 28

linkStyle default stroke:gray
```
