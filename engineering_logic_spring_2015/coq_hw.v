
(* Basics: Functional Programming in Coq *)

Inductive day : Type :=
| monday : day
| tuesday : day
| wednesday : day
| thursday : day
| friday : day
| saturday : day
| sunday : day
.

Definition next_weekday (d:day) : day :=
match d with
| monday => tuesday
| tuesday => wednesday
| wednesday => thursday
| thursday => friday
| friday => monday
| saturday => monday
| sunday => monday
end.

Eval compute in (next_weekday friday).

Example assertion_1: (next_weekday (next_weekday (next_weekday (next_weekday friday)))) = thursday.
Proof.
simpl.
reflexivity.
Qed.

Inductive bool : Type :=
| true  : bool
| false : bool
.

Definition negb (b:bool) : bool :=
match b with
| true => false
| false => true
end
.

Eval compute in (negb (negb (negb (true)))).

Definition andb (b1:bool) (b2:bool) : bool :=
match b1 with
| true => b2 
| false => false
end
.

Eval compute in (andb true true).
Eval compute in (andb true false).
Eval compute in (andb false true).
Eval compute in (andb false false).

Definition orb (b1:bool) (b2:bool) : bool :=
match b1 with
| true => true
| false => b2
end
.

Eval compute in (orb true true).
Eval compute in (orb true false).
Eval compute in (orb false true).
Eval compute in (orb false false).

Example assertion_2: (orb true true) = true.
Proof.
reflexivity.
Qed.

Example assertion_3: (orb true false) = true.
Proof.
reflexivity.
Qed.

Example assertion_4: (orb false true) = true.
Proof.
reflexivity.
Qed.

Example assertion_5: (orb false false) = false.
Proof.
reflexivity.
Qed.

Definition nandb (b1:bool) (b2:bool) : bool := (negb (andb b1 b2)).

Example assertion_6: (nandb true false) = true.
Proof.
reflexivity.
Qed.

Example assertion_7: (nandb false false) = true.
Proof.
reflexivity.
Qed.

Example assertion_8: (nandb false true) = true.
Proof.
reflexivity.
Qed.

Example assertion_9: (nandb true true) = false.
Proof.
reflexivity.
Qed.

Definition andb3 (b1:bool) (b2:bool) (b3:bool) : bool := (andb (andb b1 b2) b3).

Example assertion_10: (andb3 true true true) = true.
Proof.
reflexivity.
Qed.

Example assertion_11: (andb3 true false true) = false.
Proof.
reflexivity.
Qed.

Example assertion_12: (andb3 false true true) = false.
Proof.
reflexivity.
Qed.

Example assertion_13: (andb3 false false true) = false.
Proof.
reflexivity.
Qed.

Check true.

Check assertion_12.

Module Playground1.

Inductive nat : Type :=
| O : nat
| S : nat -> nat
.

Definition pred (n:nat) : nat :=
match n with
| O => O
| S n' => n'
end
.

End Playground1.

Check (S (S (S O))).

Check pred (S (S (S O))).

Eval compute in pred (S (S (S O))).

Definition minustwo (n:nat) : nat :=
match n with
| O => O
| S O => O
| S n' => (pred n')
end
.

Fixpoint evenb (n:nat) : bool :=
match n with
| O => true
| S O => false
| S (S n') => evenb n'
end
.

Eval compute in evenb 5.
Eval compute in evenb 6.

Definition oddb (n:nat) : bool := negb (evenb n).

Module Playground2.

Fixpoint plus (a:nat) (b:nat) : nat :=
match a with
| O => b
| S a' => plus a' (S b)
end
.

Fixpoint minus (a b:nat) : nat :=
match b with
| O => a
| S b' => minus (pred a) b'
end
.

Fixpoint mult (a b:nat) : nat :=
match a with
| O => O
| S a' => plus b (mult a' b)
end
.

Fixpoint exp (base power : nat) : nat :=
match power with
| O => S O
| S power' => mult base (exp base power')
end
.

End Playground2.

Eval compute in Playground2.plus 8 4.
Eval compute in Playground2.minus 8 55.
Eval compute in Playground2.mult 8 6.
Eval compute in Playground2.exp 2 10.

Fixpoint factorial (n : nat) : nat :=
match n with
| O => 1
| S n' => mult n (factorial n')
end
.

Eval compute in factorial 5.

Notation "x + y" := (plus x y) (at level 50, left associativity) : nat_scope.
(* coq tutorial says to not worry about level, associativity, and nat_scope for now *)

Fixpoint beq_nat (a b : nat) : bool :=
match a with
| O => match b with
| O => true
| _ => false
end
| S a' => match b with
| O => false
| S b' => beq_nat a' b'
end
end
.

Eval compute in beq_nat 9 10.
Eval compute in beq_nat 9 9.
Eval compute in beq_nat 9 8.

Fixpoint blt_nat (a b : nat) : bool :=
match a with
| O => match b with
| O => false
| S b' => true
end
| S a' => match b with
| O => false
| S b' => blt_nat a' b'
end
end
.

Eval compute in blt_nat 9 10.
Eval compute in blt_nat 9 9.
Eval compute in blt_nat 9 8.

Fixpoint ble_nat (a b : nat) : bool :=
match a with
| O => match b with
| O => true
| S b' => true
end
| S a' => match b with
| O => false
| S b' => ble_nat a' b'
end
end
.

Eval compute in ble_nat 9 10.
Eval compute in ble_nat 9 9.
Eval compute in ble_nat 9 8.

Theorem plus_O_n : forall n : nat, 0 + n = n.
Proof.
intros n.
reflexivity.
Qed.

Theorem plus_1_l : forall n:nat, 1 + n = S n.
Proof.
intros n.
reflexivity.
Qed.

Theorem mult_0_l : forall n:nat, 0 * n = 0.
Proof.
intros n.
reflexivity.
Qed.

Theorem plus_id_example : forall n m : nat, n=m -> n+n = m+m.
Proof.
intros n m.
intros H.
rewrite <- H.
reflexivity.
Qed.

Theorem plus_id_exercise : forall n m o : nat, n = m -> m = o -> n + m = m + o.
Proof.
intros n m o.
intros H.
intros J.
rewrite -> H.
rewrite <- J.
reflexivity.
Qed.

Theorem mult_0_plus : forall n m : nat, (0 + n) * m = n * m.
Proof.
intros n m.
rewrite plus_O_n.
reflexivity.
Qed.

Theorem mult_S_1 : forall n m : nat, m = S n -> m * (1 + n) = m * m.
Proof.
intros n m.
rewrite <- plus_1_l.
intros H.
rewrite <- H.
reflexivity.
Qed.

Theorem plus_1_neq_0_firsttry : forall n : nat, beq_nat (n + 1) 0 = false.
Proof.
intros n.
destruct n as [| n'].
reflexivity.
reflexivity.
Qed.

Theorem zero_nbeq_plus_1 : forall n : nat,
beq_nat 0 (n + 1) = false.
Proof.
intros n.
destruct n as [| n'].
reflexivity.
simpl.
reflexivity.
Qed.

Theorem identity_fn_applied_twice :
forall (f : bool -> bool),
(forall(x : bool), f x = x) ->
forall(b : bool), f (f b) = b.
Proof.
intros f.
intros x.
intros b.
rewrite x.
rewrite x.
reflexivity.
Qed.

Theorem andb_eq_orb :
  forall (b c : bool), (andb b c = orb b c) -> b = c.
Proof.
intros b c.
destruct b.
simpl.
destruct c.
trivial.
intros H.
inversion H.
destruct c.
trivial.
intros H.
trivial.
Qed.

(* InductionProof by Induction *)

Require Export Basics.

Require String. Open Scope string_scope.

Ltac move_to_top x :=
  match reverse goal with
  | H : _ |- _ => try move x after H
  end.

Tactic Notation "assert_eq" ident(x) constr(v) :=
  let H := fresh in
  assert (x = v) as H by reflexivity;
  clear H.

Tactic Notation "Case_aux" ident(x) constr(name) :=
  first [
    set (x := name); move_to_top x
  | assert_eq x name; move_to_top x
  | fail 1 "because we are working on a different case" ].

Tactic Notation "Case" constr(name) := Case_aux Case name.
Tactic Notation "SCase" constr(name) := Case_aux SCase name.
Tactic Notation "SSCase" constr(name) := Case_aux SSCase name.
Tactic Notation "SSSCase" constr(name) := Case_aux SSSCase name.
Tactic Notation "SSSSCase" constr(name) := Case_aux SSSSCase name.
Tactic Notation "SSSSSCase" constr(name) := Case_aux SSSSSCase name.
Tactic Notation "SSSSSSCase" constr(name) := Case_aux SSSSSSCase name.
Tactic Notation "SSSSSSSCase" constr(name) := Case_aux SSSSSSSCase name.


Theorem andb_true_elim1 : forall b c : bool,
  andb b c = true -> b = true.
Proof.
  intros b c. 
  intros H.
  destruct b.
  Case "b = true". (* <----- here *)
    reflexivity.
  Case "b = false". (* <---- and here *)
    rewrite <- H.
    reflexivity.
Qed.

Theorem andb_true_elim2 : forall b c : bool,
  andb b c = true -> c = true.
Proof.
intros b c.
intros H.
destruct c.
  Case "c = true".
    reflexivity.
  Case "c = false".
    rewrite <- H.
     destruct b.
       SCase "b = true".
         reflexivity.
       SCase "b = false". 
         reflexivity.
Qed.

Theorem plus_0_r_firsttry : forall n:nat,
  n + 0 = n.
Proof.
intros n.
induction n as [|n'].
  Case "n = 0".
    reflexivity.
  Case "n = S n'".
    simpl.
    rewrite -> IHn'.
    reflexivity.    
Qed.

Theorem minus_diag : forall n,
  minus n n = 0.
Proof.
intros n.
induction n as [|n'].
  Case "n=0".
    simpl.
    reflexivity.
  Case "n=S n'".
    simpl.
    rewrite -> IHn'.
   reflexivity.    
Qed.

Theorem mult_0_r : forall n:nat,
  n * 0 = 0.
Proof.
intros n.
induction n as [|n'].
  Case "n = 0".
     simpl.
     reflexivity.
  Case "n = S n'".
    simpl.
    rewrite -> IHn'.
    reflexivity.
Qed.

Theorem plus_n_Sm : forall n m : nat,
  S (n + m) = n + (S m).
Proof.
intros n m.
induction n as [|n'].
  Case "n = 0".
    simpl.
    reflexivity.
  Case "n = S n'".
    simpl.
    rewrite -> IHn'.
    reflexivity.
Qed.

Theorem plus_comm : forall n m : nat,
  n + m = m + n.
Proof.
intros n m.
induction n as [|n'].
  Case "n = 0".
    simpl.
    induction m as [|m'].
      SCase "m = 0".
        simpl.
        reflexivity.
      SCase "m = S m'".
        simpl.
        rewrite <- IHm'.
        reflexivity.
  Case "n = S n'".
    induction m as [|m'].
      SCase "m = 0".
        simpl.
        rewrite -> IHn'.
        simpl.
        reflexivity.
      SCase "m = S m'".
        simpl.
        rewrite -> IHn'.
        simpl.
        assert (S (m' + n') = (m' + S n')).
          apply plus_n_Sm.
        rewrite -> H.
        reflexivity.
Qed.

Theorem plus_assoc : forall n m p : nat,
  n + (m + p) = (n + m) + p.
Proof.
intros n m p.
induction n as [|n'].
  Case "n = 0".
    simpl.
    reflexivity.
  Case "n = S n'".
    induction m as [|m'].
      SCase "m = 0".
        simpl.
        rewrite <- IHn'.
        simpl.
        reflexivity.
      SCase "m = S m'".
        simpl.
        rewrite <- IHn'.
        simpl.
        reflexivity.
Qed.

Fixpoint double (n:nat) :=
match n with
  | O => O
  | S n' => S (S (double n'))
end.

Lemma double_plus : forall n, double n = n + n .
Proof.
intros n.
induction n as [|n'].
  Case "n = 0".
    simpl.
    reflexivity.
  Case "n = S n'".
    simpl.
    assert ((S n' + n') =(n' + S n')).
       apply plus_n_Sm.
    rewrite -> IHn'.
    rewrite <- H.
    simpl.
    reflexivity.
Qed.

Theorem plus_swap : forall n m p : nat,
  n + (m + p) = m + (n + p).
Proof.
intros n m p.
assert (A: n+(m+p) = (n+m)+p) by apply plus_assoc.
rewrite -> A.
assert (B: m+(n+p)=(m+n)+p) by apply plus_assoc.
rewrite -> B.
assert (C: m+n=n+m) by apply plus_comm.
rewrite -> C.
reflexivity.
Qed.

Theorem mult_comm : forall m n : nat,
 m * n = n * m.
Proof.
intros n m.
induction n as [|n'].
  Case "n = 0".
    simpl.
    assert (m*0 = 0).
      apply mult_0_r.
      rewrite -> H.
    reflexivity.
  Case "n = S n' ".
    simpl.
    assert (A: forall a b : nat ,  a + a * b = a * S b).
      intros a b.
      induction a as [|a'].
        SCase "a=0". reflexivity.
        SCase "a = S a'".
        simpl.
        rewrite <- IHa'.
        rewrite <- plus_swap.
        reflexivity.
   rewrite <- A. 
   rewrite -> IHn'.
   reflexivity.
Qed.

Theorem plus_rearrange : forall n m p q : nat,
  (n + m) + (p + q) = (m + n) + (p + q).
Proof.
intros n m p q.
assert (H: n + m = m + n).
  Case "Proof of assertion".
    rewrite -> plus_comm. reflexivity.
rewrite -> H. reflexivity. 
Qed.

Theorem evenb_n__oddb_Sn : forall n : nat,
  evenb n = negb (evenb (S n)).
Proof.
intros n.
induction n as [|n'].
  Case "n=0".
    simpl. trivial.
  Case "n = S n'".
    assert (evenb n' = evenb (S (S n'))) as H.
      simpl.
      trivial.
    rewrite <- H.
    rewrite -> IHn'.
    assert (forall s, s = negb (negb s)) as H1.
      intros s.
      destruct s.
        SCase "s=true".
          simpl. trivial.
        SCase "s=false".
          simpl. trivial.
    rewrite <- H1.
    trivial.
Qed.

(* Lists: Working with Structured Data *)

(* Require Export Induction. *)

Module NatList.

Inductive natprod : Type := 
  | pair : nat -> nat -> natprod
.

Check (pair 3 5).

Definition fst (p:natprod) : nat :=
  match p with 
    | pair a b => a
  end
.

Definition snd (p:natprod) : nat :=
  match p with 
    | pair a b => b
  end
.

Compute fst (pair 1 2).
Compute snd (pair 1 2).

Notation "( x , y )" := (pair x y).

Compute fst (1, 2).
Compute snd (1, 2).

Definition fst' (p:natprod) : nat :=
  match p with 
    | (a,b) => a
  end
.

Definition snd' (p:natprod) : nat :=
  match p with 
    | (a,b) => b
  end
.

Definition swap_pair (p:natprod) : natprod :=
  match p with 
    | (a,b) => (b,a)
  end
.

Compute fst' (1, 2).
Compute snd' (3, 4).
Compute fst' (pair 5 6).
Compute snd' (pair 7 8).
Compute swap_pair (pair 9 10).
Compute swap_pair (11,12).

Theorem surjective_pairing' : forall (n m : nat), (n,m) = (fst (n,m), snd (n,m)).
Proof.
intros n m.
simpl.
trivial.
Qed.

Theorem surjective_pairing : forall (p : natprod), p = (fst p, snd p).
Proof.
intros p.
destruct p.
simpl.
trivial.
Qed.

Theorem snd_fst_is_swap : forall (p : natprod), (snd p, fst p) = swap_pair p.
Proof.
intros p.
destruct p.
simpl.
trivial.
Qed.

Theorem fst_swap_is_snd : forall (p : natprod), fst (swap_pair p) = snd p.
Proof.
intros p.
destruct p.
simpl. trivial.
Qed.

Inductive natlist : Type :=
  | nil : natlist
  | cons : nat -> natlist -> natlist
.

Definition mylist := cons 1 (cons 2 (cons 3 nil)).

Check mylist.
Compute mylist.

Notation " x :: l " := (cons x l) (at level 60, right associativity).
Notation " [] " := (nil).
Notation " [ a ; .. ; b ] " := (cons a .. (cons b nil) .. ).

Fixpoint repeat (n count : nat) : natlist :=
  match count with 
    | O => []
    | S count' => n :: (repeat n count')
  end
.

Compute (repeat 4 5).

Fixpoint length (l:natlist) : nat := 
  match l with 
    | nil => O
    | h :: t => S (length t)
  end
.

Compute (length (repeat 1 2)).

Fixpoint app (l1 l2 : natlist) : natlist := 
  match l1 with
    | nil => l2
    | h :: t => h :: (app t l2)
  end
.

Compute (app [4;5;6] (repeat 1 2)).

Notation "a ++ b" := (app a b) (right associativity, at level 60).

Compute ([4;5;6] ++ [7;8]).

Definition hd (default : nat) (l : natlist) : nat :=
  match l with 
    | nil => default
    | h :: t => h
  end
.

Definition tl (l : natlist) : natlist :=
  match l with 
    | nil => nil
    | h :: t => t
  end
.

Compute hd 99 [4;5;6].
Compute hd 99 [].
Compute tl [].
Compute tl [4;5;6].

Fixpoint nonzeros (l:natlist) : natlist :=
  match l with
    | nil => nil
    | h :: t => match h with 
                  | O => nonzeros t
                  | S n => h :: nonzeros t
                end
  end
.

Compute nonzeros [0;1;0;2;3;0;0].

Fixpoint oddmembers (l:natlist) : natlist :=
  match l with 
    | nil => nil
    | h :: t => 
      match (oddb h) with 
        | true => h :: (oddmembers t) 
        | false => (oddmembers t)
      end
  end
.

Compute oddmembers [0;1;0;2;3;0;6;0].

Definition countoddmembers (l:natlist) : nat :=
  length (oddmembers l)
.

Compute countoddmembers [1;0;3;1;4;5].
Compute countoddmembers [0;2;4].
Compute countoddmembers nil.

Fixpoint alternate (l1 l2 : natlist) : natlist :=
  match l1 with
    | nil => l2
    | h1 :: t1 => 
      match l2 with 
        | nil => l1
        | h2 :: t2 => h1 :: h2 :: (alternate t1 t2)
      end
  end
.

Compute alternate [1;2;3] [4;5;6].
Compute alternate [1] [4;5;6].
Compute alternate [1;2;3] [4].
Compute alternate [] [20;30].

Definition bag := natlist.

Fixpoint count (v:nat) (s:bag) : nat := 
  match s with
    | nil => 0
    | h :: t => 
      match (beq_nat v h) with
        | true => 1 + (count v t)
        | _ => (count v t)
      end
  end
.

Compute count 1 [1;2;3;1;4;1].
Compute count 6 [1;2;3;1;4;1].

Definition sum : bag -> bag -> bag := app.

Compute count 1 (sum [1;2;3] [1;4;1]).

Definition add (v:nat) (s:bag) : bag := ([v] ++ s).

Compute count 1 (add 1 [1;4;1]).
Compute count 5 (add 1 [1;4;1]).

Definition member (v:nat) (s:bag) : bool := blt_nat 0 (count v s).

Compute member 1 [1;4;1].
Compute member 2 [1;4;1].

Fixpoint remove_one (v:nat) (s:bag) : bag := 
  match s with 
    | nil => nil
    | h :: t => 
      match (beq_nat h v) with
        | true => t
        | false => h :: (remove_one v t)
      end
  end 
. 

Compute count 5 (remove_one 5 [2;1;5;4;1]).
Compute count 5 (remove_one 5 [2;1;4;1]).
Compute count 4 (remove_one 5 [2;1;4;5;1;4]).
Compute count 5 (remove_one 5 [2;1;5;4;5;1;4]).

Fixpoint remove_all (v:nat) (s:bag) : bag :=
  match s with 
    | nil => nil
    | h :: t => 
      match (beq_nat h v) with
        | true => (remove_all v t)
        | false => h :: (remove_all v t)
      end
  end 
. 

Compute count 5 (remove_all 5 [2;1;5;4;1]).
Compute count 5 (remove_all 5 [2;1;4;1]).
Compute count 4 (remove_all 5 [2;1;4;5;1;4]).
Compute count 5 (remove_all 5 [2;1;5;4;5;1;4;5;1;4]).

Fixpoint subset (s1:bag) (s2:bag) : bool :=
  match s1 with
  | nil => true
  | h1 :: t1 =>
    match (member h1 s2) with
    | true => subset t1 (remove_one h1 s2)
    | false => false
    end
  end
.

Compute subset [1;2] [2;1;4;1].
Compute subset [1;2;2] [2;1;4;1].

Theorem nil_app : forall l:natlist, [] ++ l = l.
Proof. 
intros. 
simpl. 
trivial.
Qed.

Theorem tl_length_pred : forall l:natlist, pred (length l) = length (tl l).
Proof.
intros.
simpl.
destruct l.
  Case "l = nil".
    simpl.
    trivial.
  Case "l = n :: l".
    simpl.
    trivial.
Qed.

Theorem app_assoc : forall l1 l2 l3 : natlist, (l1 ++ l2) ++ l3 = l1 ++ (l2 ++ l3).
Proof.
intros. 
induction l1.
  Case "l1 = nil".
    simpl. 
    trivial.
  Case "l1 = n :: l1".
    simpl.
    rewrite -> IHl1.
    trivial.
Qed.

Theorem app_length : forall l1 l2 : natlist, length (l1 ++ l2) = (length l1) + (length l2).
Proof.
intros.
induction l1.
  Case "l1 = nil".
    simpl.
    trivial.
  Case "l1 = n :: l1".
    simpl.
    rewrite <- IHl1.
    trivial.
Qed.

Fixpoint snoc (l:natlist) (v:nat) : natlist :=
match l with
  | nil => [v]
  | h :: t => h :: (snoc t v)
end.

Fixpoint rev (l:natlist) : natlist := 
match l with
  | nil => []
  | h :: t => snoc (rev t) h
end.

Compute rev [1;2;3].

Theorem length_snoc : forall n : nat, forall l : natlist, length (snoc l n) = S (length l).
Proof.
intros.
induction l.
Case "l = nil".
  simpl.
  trivial.
Case "l = (n :: l)".
  simpl.
  rewrite -> IHl.
  trivial.
Qed.

Theorem rev_length : forall l : natlist, length (rev l) = length l.
Proof.
intros.
induction l.
Case "l = nil".
  compute.
  trivial.
Case "l = (n :: l)".
  simpl.
  rewrite <- IHl.
  apply length_snoc.
Qed.

Theorem app_nil_end : forall l : natlist, l ++ [] = l.
Proof.
intros.
induction l.
Case "l = nil".
  compute. trivial.
Case "l = (n :: l)".
  simpl.
  rewrite IHl.
  trivial.
Qed.

Theorem rev_involutive : forall l : natlist, rev (rev l) = l.
Proof.
intros.
induction l.
Case "l = nil".
  compute. trivial.
Case "l = (n :: l)".
  simpl.
Admitted.

Theorem app_assoc4 : forall l1 l2 l3 l4 : natlist, l1 ++ (l2 ++ (l3 ++ l4)) = ((l1 ++ l2) ++ l3) ++ l4.
Proof.
intros.
assert (l1 ++ l2 ++ l3 = ((l1 ++ l2) ++ l3) ) as H0.
  rewrite <- app_assoc.
  trivial.
rewrite <- H0.
assert (l2 ++ l3 ++ l4 = ((l2 ++ l3) ++ l4) ) as H1.
  rewrite <- app_assoc.
  trivial.
rewrite -> H1.
rewrite <- app_assoc.
trivial.
Qed.

Theorem snoc_append : forall (l:natlist) (n:nat), snoc l n = l ++ [n].
Proof.
intros.
induction l.
Case "l = nil".
  compute. trivial.
Case "l = (n :: l)".
  simpl.
  rewrite -> IHl.
  trivial.
Qed.

Theorem distr_rev : forall l1 l2 : natlist, rev (l1 ++ l2) = (rev l2) ++ (rev l1).
Proof.
intros.
induction l1.
Case "l1 = nil".
  induction l2.
  SCase "l2 = nil".
    compute. trivial.
  SCase "l2 = (n :: l2)".
    rewrite -> app_nil_end.
    simpl.
    trivial.
Case "l1 = (n :: l)".
  induction l2.
  SCase "l2 = nil".
    simpl.
    rewrite -> app_nil_end.
    trivial.
  SCase "l2 = (n :: l2)".
    simpl.
    rewrite -> IHl1.
    rewrite snoc_append.
    rewrite snoc_append.
    rewrite snoc_append.
    simpl.
    rewrite snoc_append.
    apply app_assoc.
Qed.

Lemma nonzeros_app : forall l1 l2 : natlist, nonzeros (l1 ++ l2) = (nonzeros l1) ++ (nonzeros l2).
Proof.
intros.
induction l1.
Case "l1 = nil".
  induction l2.
  SCase "l2 = nil".
    reflexivity.
  SCase "l2 = (n :: l2)".
    reflexivity.
Case "l1 = (n :: l)".
  destruct n.
  SCase "n = O".
    simpl. 
    rewrite IHl1.
    trivial.
  SCase "n = S n".
    simpl.
    rewrite IHl1.
    trivial.
Qed.

Fixpoint beq_natlist (l1 l2 : natlist) : bool :=
match l1, l2 with 
  | nil, nil => true
  | h1::t1, h2::t2 => 
    match (beq_nat h1 h2) with
      | true => (beq_natlist t1 t2)
      | false => false
    end
  | _, _ => false
end
.

Compute beq_natlist nil nil.
Compute beq_natlist [1;2;3] [1;2;3].
Compute beq_natlist [1;2;3] [1;2;4].

Theorem beq_natlist_refl : forall l:natlist, true = beq_natlist l l.
Proof.
intros.
induction l.
Case "l = nil".
  simpl. trivial.
Case "l = (n :: l)".
  simpl.
  assert (beq_nat n n = true) as H.
    induction n.
    SCase "n = O".
      simpl. trivial.
    SCase "n = S n".
      simpl.
      rewrite IHn.
      trivial.
  rewrite H.
  rewrite IHl.
  trivial.
Qed.

Theorem count_member_nonzero : forall (s : bag), ble_nat 1 (count 1 (1 :: s)) = true.
Proof.
intros.
induction s.
Case "s = [1]".
  simpl. trivial.
Case "s = 1::n::s".
(*
  simpl.
  induction n.
  SCase "n = O".
    compute.
    
  simpl.
  
induction s.
Case "s = [1]".
  simpl. trivial.
Case "s = 1::n::s".
  destruct n.
  SCase "n = O".
    simpl.
  simpl.
Qed.
*)
Admitted.

Theorem ble_n_Sn : forall n, ble_nat n (S n) = true.
Proof. 
intros.
induction n.
simpl. trivial.
simpl.
rewrite IHn.
trivial.
Qed.

Theorem remove_decreases_count: forall (s : bag), ble_nat (count 0 (remove_one 0 s)) (count 0 s) = true.
Proof.
intros.
induction s.
  Case "s = []".
    reflexivity.
  Case "s = n::s".
    induction n.
      SCase "n = O".
        simpl.
        rewrite ble_n_Sn.
        trivial.
      SCase "n = S n".
        simpl.
        rewrite IHs.
        trivial.
Qed.


Theorem rev_injective : forall (l1 l2 : natlist), rev l1 = rev l2 -> l1 = l2. 
Proof.
intros.
rewrite rev_involutive.
assert 
Qed.











Theorem and_example :
  (0 = 0) /\ (4 = mult 2 2).
Proof.
  apply conj.
  Case "left". 
    reflexivity.
  Case "right". 
    reflexivity. 
Qed.

Theorem proj1 : forall P Q : Prop,
  P /\ Q -> P.
Proof.
  intros P Q H.
  destruct H as [HP HQ].
  apply HP. 
Qed.

Theorem iff_sym : forall P Q : Prop,
  (P <-> Q) -> (Q <-> P).
Proof.
  intros P Q H.
  destruct H as [HAB HBA].
  split.
    Case "→". apply HBA.
    Case "←". apply HAB. 
Qed.

Theorem iff_refl : forall P : Prop,
  P <-> P.
Proof.
intros.
apply conj.
    Case "->". trivial.
    Case "<-". trivial.
Qed.


Theorem or_distributes_over_and_1 : forall P Q R : Prop,
  P \/ (Q /\ R) -> (P \/ Q) /\ (P \/ R).
Proof.
  intros P Q R. intros H. destruct H as [HP | [HQ HR]].
    Case "left". split.
      SCase "left". left. apply HP.
      SCase "right". left. apply HP.
    Case "right". split.
      SCase "left". right. apply HQ.
      SCase "right". right. apply HR. 
Qed.

Theorem or_distributes_over_and_2 : forall P Q R : Prop,
  (P \/ Q) /\ (P \/ R) -> P \/ (Q /\ R).
Proof.
intros.
inversion H.
inversion H0.
left.
apply H2.
inversion H1.
left.
apply H3.
right.
apply conj.
apply H2.
apply H3.
Qed.

Theorem andb_false : forall b c,
  andb b c = false -> b = false \/ c = false.
Proof.
intros.
destruct H as [B C].
left.















