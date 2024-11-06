## Linear Knapsack Problem (LKP)

### Problem Statement
Consider the Linear Knapsack Problem (LKP):

$$
\text{Maximize } f(x) = \sum_{i=1}^{16} v_i x_i
$$

subject to

$$
\sum_{i=1}^{16} w_i x_i \leq W
$$

where \( x^T = (x_1, x_2, \dots, x_{16}) \), with \( n = 16 \), and \( x_i \) are the binary optimization variables. The data for the problem is given as:

\[
(v_i, w_i) = \{(6, 3), (8, 5), (3, 4), (4, 7), (5, 4), (9, 10), (11, 3), (12, 6), (6, 8), (8, 14), (13, 4), (15, 9), (16, 10), (13, 11), (9, 17), (25, 12)\}
\]

with total capacity \( W = 25 \).

### Question 1: Penalty Function for GA Solution

To solve the above Knapsack Problem, construct the following penalty function:

$$
F(x) = f(x) - R \cdot \epsilon(x)
$$

where \( R \) is the penalty parameter, and \( \epsilon(x) = \max\left(0, \sum_{j=1}^{16} w_j x_j - W\right) \).

Solve the problem using a binary-coded Genetic Algorithm (GA) with the penalty function \( F(x) \) as the fitness function. The parameters for GA are:

- Crossover probability \( p_c = 1 \)
- Mutation probability \( p_m = 10^{-3} \)
- Population size \( N = 20 \)
- Children per iteration \( m = 6 \)
- Maximum iterations = 30
- Penalty parameter \( R = 50 \)

### Question 2: Local Search Using the Fiduccia-Mattheyses (FM) Algorithm

Solve the above problem using the FM local search algorithm. The initial solution \( x_0 = (x_1, x_2, \dots, x_{16}) \) is defined such that \( x_i = 0 \) if \( i \) is even, and \( x_i = 1 \) if \( i \) is odd. 

The FM algorithm steps are as follows:

1. Initialize \( x_0 \), calculate \( f(x_0) \), set \( x_{\text{max}} = x_0 \), flag = 1, Pass = 1.
2. Let \( F \) be the set of indices for unlocked/free variables (initially \( F = \{1, 2, \dots, n\} \)), and \( L \) the set of locked variables (initially \( L = \emptyset \)).

3. **While** flag = 1:
   - Set flag = 0, Epoch = 0, \( F = \{1, 2, \dots, n\} \), \( L = \emptyset \).
   - Repeat until Epoch = n:
      - For each \( j \in F \), calculate \( x_j \) by flipping \( x_j \) in \( x_t \) (use \( x_0 \) for Epoch = 0), then find \( f(x_j) \).
      - Let \( x_t = \arg \max_{j \in F} \{f(x_j)\} \). Update \( F = F \setminus \{t\} \), \( L = L \cup \{t\} \), and Epoch = Epoch + 1.
   - Let \( x_t \) be the best solution at the end of Epoch \( t \), and \( x = \arg \max_{t \in L} f(x_t) \).
   - If \( f(x) > f(x_0) \):
      - Update \( x_{\text{max}} = x \), \( x_0 = x \), flag = 1, Pass = Pass + 1.
4. End While.
5. Return \( x_{\text{max}} \).
