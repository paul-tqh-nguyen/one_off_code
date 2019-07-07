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

# Naive Exhaustive Solution

We can do an exhaustive O(n**2) search to ensure that we get the correct answer.

Every possible solution involves picking two walls.

There are n walls.

There are (n choose 2) possible solutions.

We can just calculate all of the areas for all the possible solutions (takes O(n choose 2) time) and pick the max (takes O(n choose 2) time).

(n choose 2) = n * (n-1) / 2.

Thus, O(n choose 2) = O(n * (n-1) / 2) = O(n**2).

