This is the write up for an algorithm presented at this reddit post: https://www.reddit.com/r/csinterviewproblems/comments/all9xv/walls_and_water_problem/

This is just a write up of the proof we came up with.

The nice plan is to eventually have it such that we have a nice tutorial via REACT that'll create random examples and show that the algorithm works everytime.

# Intro

Here's the problem as it is stated:

```
There are 'n' walls and distance between consecutive walls is 1 unit. And the list describing walls' heights 'H' is given. We have to select only two walls out of given 'n' walls which can contain the maximum amount of water. Indices of the selected two walls and the water stored in between them should be printed as output.

e.g: n = 6 H = [1, 3, 5, 3, 5, 3] For this case, output is; indices = (1, 5) water = min(H[1], H[5]) * (indices[1] - indices[0]) = 3 * 4 = 12

Brute force approach for this problem is not the efficient approach. Any other optimised approach of solving this is appreciated. I really need the solution to this as I've missed out on quite a few jobs because of this problem. Thank you.
```

# Naive Exhaustive O(n**2) Solution

We can do an exhaustive O(n**2) search to ensure that we get the correct answer.

Every possible solution involves picking two walls.

There are n walls.

There are (n choose 2) possible solutions.

We can just calculate all of the areas for all the possible solutions (takes O(n choose 2) time) and pick the max (takes O(n choose 2) time).

(n choose 2) = n * (n-1) / 2.

Thus, O(n choose 2) = O(n * (n-1) / 2) = O(n**2).

# O(n log n) Solution (possibly unnecessarily verbose)

Here's the first more-optimal-than-an-exhaustive-search algorithm we came up with.

It can definitely do with some cleaning up, refactoring, and simplification for easier explanation.

Let's call our input list of walls input_walls.

Let's create a struct for each wall that contains the wall's height and original position in input_walls as attributes. Let's call this class wall_struct. 

Let's create a list of wall_struct instances from input_walls. Let's call this list wall_structs.

Let's sort wall_structs by two criterion, the height attribute first and then the original position as a tie breaker (later position in input_walls means you go later in this sorting). 

Let's create a new struct called possible_solution. It has 5 attributes that are the starting position, height at the starting position, ending position, height at the ending position, and area.

Let's create a trivial possible_solution for each wall_struct in the now sorted wall_structs list. A trivial possible_solution can be created from a wall_truct instance by having the starting and ending positions be the same (the position of the wall_struct), teh starting and ending heights be the stame (the height of the wall struct), and the area be zero.

Let's create a function that takes two possible solutions and merges them to create a more optimal solution from the both of them.

Here's how it would work:
* There are 4 walls among the two possible solutions.
* Are (4 choose 2) = 6 (i.e. a constant number of) ways we can create a solution from the two.
* We can create the 6 new possible solutions structs and calculate their areas.
* Find the one with the largest area (this takes O(6) = O(1) time).
* We have our new optimal solution.

We can do a merge-sort-style divide-and-conquer technique here to merge all the trivial possible solutions into one optimal solution.

We go n merges (log n) times to get the optimal solution.

This whole algorithm takes O(n log n) time. 

# O(n log n) Solution (possibly incorrect)

Take the divide-and-conquer and merge-sort style of the approach above, and just apply it to the list right out the gate (in particular, the same thing without the sorting of the wall_struct instances). 

I'm not sure if this is correct though. This is not algorithmically faster but is faster, but it is harder to prove correctness. 
