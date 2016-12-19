(* x = y under liebniz equality, but not some other semantically significant equality *)

Class EqDec (A : Type) := {
  eqb : A -> A -> bool ;
  eqb_eq : forall x y: A, eqb x y = true -> x = y
}.

Inductive money  : Type :=
  | one_dollar : money 
  | four_quarters : money
.

Definition beq_unit (a b :unit) : bool := true.

Theorem theorem_0: 
  forall  x y: unit, beq_unit x y = true -> x = y.
Proof.
intros x y.
intros H.
destruct x.
destruct y.
reflexivity.
Qed.

Definition money_eq (a b :money) : bool := true.

Theorem theorem_1: 
  forall x y: money, money_eq x y = true -> x = y.
Proof.
intros x y.
intro H.
destruct x.
destruct y.
trivial.
Abort.

Instance unit_EqDec : EqDec unit := {
  eqb := beq_unit;
  eqb_eq := theorem_0
}.


Definition beq_bool (a b :bool) : bool := 
match a with
  | true => match b with 
              | true => true
              | false => false
            end
  | false => match b with 
               | false => true
               | true => false
             end
end
.

Theorem theorem_2: 
  forall x y: bool, beq_bool x y = true -> x = y.
Proof.
intros x y.
intros H.
destruct x.
destruct y.
trivial.
rewrite <- H.
simpl.
trivial.
destruct y.
rewrite <- H.
simpl.
trivial.
trivial.
Qed. 

Instance bool_EqDec : EqDec bool := {
  eqb := beq_bool;
  eqb_eq := theorem_2
}.

Eval compute in eqb true true.
Eval compute in eqb false true.
Eval compute in eqb true false.
Eval compute in eqb false false.

Require Export Basics.
Require Import List.
Import ListNotations.

Eval compute in [1;2;3].

Fixpoint beq_nat (n m : nat) : bool :=
  match n with
  | O => match m with
         | O => true
         | S m' => false
         end
  | S n' => match m with
            | O => false
            | S m' => beq_nat n' m'
            end
  end.

Fixpoint beq_nat_list (a b :(list nat)) : bool := 
match a,b with
  | nil, nil => true
  | nil, h_b::t_b => false
  | h_a::t_a, nil => false
  | h_a::t_a, h_b::t_b => andb (beq_nat h_a h_b) (beq_nat_list t_a t_b)
end.

Eval compute in beq_nat_list [1;9] [1;9].
Eval compute in beq_nat_list [2;9] [1;9].
Eval compute in beq_nat_list [2;9] [8;9;9].
Eval compute in beq_nat_list [6;2;9] [8;9;9].

(* 

Instance nat_list_EqDec : EqDec (list nat) := {
  eqb := beq_nat_list
}.

Proof.
intros x y.
intro.
induction (beq_nat_list x y).
Abort.

*)

(******************************************************************************************)

Record Circle : Set :=
  circle {
    circle_x_coord: nat;
    circle_y_coord: nat;
    circle_radius: nat
  }
.

Record Rectangle : Set := 
  rectangle {
    rectangle_x_coord: nat;
    rectangle_y_coord: nat;
    rectangle_height: nat;
    rectangle_width: nat
  }
.

Eval compute in (circle 3 2 1).(circle_x_coord).
Eval compute in (circle 3 2 1).(circle_y_coord).
Eval compute in (circle 3 2 1).(circle_radius).

(*
Inductive Shape : Set := 
  | C : Circle -> Shape 
  | R : Rectangle -> Shape 
. 

Definition s1: Shape := C(circle 0 0 1).
Definition s2: Shape := R(rectangle 0 0 1 2).

Class Area (A:Type) := {
  area: A-> nat;
  pos_area: forall a:A, area a >0
}.

Instance Circle_area : Area Circle := {
  (*
  area := fun c: Circle => match c with
                             | circle xx yy rr => 3 * rr * rr
                           end
  *)
  area := fun c: Circle => 3
}.
Proof.
intro.
auto.
Qed.

Instance Rectangle_area : Area Rectangle := {
  (*
  area := fun r: Rectangle => match r with
                                | rectangle xx yy hh ww => hh * ww
                              end
  *)
  area := fun r: Rectangle => 5
}.
Proof.
intro.
do 4 apply le_S.
auto.
Qed.

Instance Shape_area : Area (Shape) := {
  area := fun ss => 
    match ss with 
      | C cc => area cc
      | R rr => area rr
    end
}.

Eval compute in area (C (circle 0 0 9)). 
Eval compute in area (R (rectangle 0 0 3 4)).

Definition non_zero_area_circles : {c:Circle | c.(circle_x_coord)>0}.
Definition non_zero_area_rectangles : {r:Rectangle | r.(rectangle_height)>0 /\ r.(rectangle_width)>0}.

Definition positive_nat := {k: nat | k >0}.
Check positive_nat.

Print sig.
Theorem nine_is_gte_zero : 9 > 0.
Proof.
Admitted.
Example pn: positive_nat.
  apply (exist _ 1). auto.
*)

Class Shape (A:Type) := {
  area: A-> nat;
  pos_area: forall a:A, area a >0
}.

Instance Circle_area : Shape Circle := {
  area := fun c: Circle => 3
}.
Proof.
intro.
auto.
Qed.

Instance Rectangle_area : Shape Rectangle := {
  area := fun r: Rectangle => 5
}.
Proof.
intro.
do 4 apply le_S.
auto.
Qed.

Definition anN:= 3.
Theorem anNNotO: 3<>0.
Proof.
intro.
inversion H.
Qed.

(* the (fun n) code below is a property, i.e. func from type instance to a Prop about it *)
Definition n : {k: nat | k<>0} := exist (fun n: nat => n<>0) 3 anNNotO.

Definition goodPred (n:{k:nat | k<>0}) : nat :=
  match n with
    exist v P => pred v
  end.

Compute goodPred n.

Extraction Language Haskell.
Recursive Extraction goodPred.

Definition positive_nat := {k: nat | k >0}.
Check positive_nat.

Print sig.
Theorem nine_is_gte_zero : 9 > 0.
Proof.
Admitted.
Example pn: positive_nat.
  apply (exist _ 1). auto.

(***********************************************************)

Import ListNotations.
(*
Class Functor (f : Type -> Type) := {
  fmap : forall a b, (a -> b) -> (f a) -> (f b)
}.

Fixpoint map_list (A B : Type) (f : A -> B) (l : (list A)) : (list B) := 
match l with
  | [] => []
  | h :: t => (f h) :: (map_list A B f t)
end.

Instance list_functor : Functor (list) := {
  fmap A B := (fix f_map_list (f : A -> B) (l : (list A)) : (list B) := 
    match l with
      | [] => []
      | h :: t => (f h) :: (f_map_list f t)
    end )
}.

Eval compute in (fun a:nat => a*a) 5.
Eval compute in fmap nat nat (fun a:nat => a*a) [1;2;3;4;5].

Instance option_functor : Functor (option) := {
  fmap A B := (fun (f : A -> B) (container : (option A)) =>
                 match container with
                   | None => None
                   | Some e => Some (f e)
                 end
              ) 
}.

Eval compute in (fun a:nat => a*a) 5.
Eval compute in fmap nat nat (fun a:nat => a*a) (Some 6).
Eval compute in fmap nat nat (fun a:nat => a*a) (None).
*)

(*
Class Monoid (f: Type -> Type) (e: Type) := {
  mempty : (f e);
  mappend : (f e) -> (f e) -> (f e);
  mconcat : (list (f e)) -> (f e) 
}.

Fixpoint nat_list_append (a b : (list nat)) : (list nat) := 
match a with
  | [] => b
  | h :: t => (h) :: (nat_list_append t b)
end.

Fixpoint foldr {X Y:Type} (f: X->Y->Y) (b:Y) (l:list X) : Y :=
  match l with
  | nil => b
  | h :: t => f h (foldr f  b t) 
  end
.

Eval compute in nat_list_append [1;2;3] [4;5;6].

Instance nat_list_monoid : Monoid (list) (nat) := { 
  mempty := [];
  mappend := nat_list_append;
  mconcat := (foldr nat_list_append [])
}.

Eval compute in mappend [1;2;3] [4;5;6].
Eval compute in mappend mempty [4;5;6].
Eval compute in mappend [1;2;3] mempty.
Eval compute in [[1;2;3];[4;5;6];[7;8;9]].
Eval compute in mconcat [[1;2;3];[4;5;6];[7;8;9]].
*)

(*******************************************************************)

Class Monoid (e: Type) := {
  mempty : e;
  mappend : e -> e -> e;
  assoc : forall a b c: e, (mappend (mappend a b) c) = (mappend a (mappend b c));
  ida : forall a: e, ((mappend a mempty) = a) /\ (a = (mappend mempty a))
}.

Fixpoint foldr {A: Type} (m: Monoid A) (l: list A): A := 
  match l with
    | [] => mempty
    | h :: t => mappend h (foldr m t) 
  end
.

Fixpoint foldl {A: Type} (m: Monoid A) (l: list A): A := 
  match l with
    | [] => mempty
    | h :: t => mappend (foldl m t) h
  end
.

Require Import Arith.
SearchAbout plus.

Instance nat_monoid_plus : Monoid (nat) := { 
  mempty := 0;
  mappend := plus
}.
Proof.
intros.
symmetry.
apply plus_assoc.
intros a.
apply conj.
assert (a + 0 = 0 + a) as H by apply plus_comm.
symmetry.
rewrite -> H.
simpl.
trivial.
simpl.
trivial.
Qed.

Eval compute in foldr nat_monoid_plus [].
Eval compute in foldr nat_monoid_plus [1;2;3].

(*
Instance nat_monoid_mult : Monoid (nat) := { 
  mempty := 1;
  mappend := mult
}.

Eval compute in foldr nat_monoid_mult [7;8].
Eval compute in foldr nat_monoid_mult [].

Instance list_of_lists_monoid : Monoid (list nat) := { 
  mempty := [];
  mappend := (fix list_concat (a b: (list nat)) := 
    match a with
      | [] => b
      | h :: t => h :: (list_concat t b)
    end)
}.

Eval compute in foldr list_of_lists_monoid [[1;2];[1;2];[3;4];[3;4]].

Instance nat_list_monoid : Monoid (list nat) := { 
  mempty := [];
  mappend := (fix nat_list_append (a b : (list nat)) :=
                match a with
                  | [] => b
                  | h :: t => (h) :: (nat_list_append t b)
                end
             )
}.

Eval compute in mempty.

Eval compute in mappend [1;2;3] [4;5;6].
Eval compute in mappend mempty [4;5;6].
Eval compute in mappend [1;2;3] mempty.
*)

Definition foo: {0=0}+{1=0}.
apply left.
trivial.
Defined.

Definition foo': {0=0}+{0<>0}.
apply left.
trivial.
Defined.

Print foo'.

(*
Check foo''.

Compute foo'' 3 4.
Compute foo'' 3 3.

Check left eq_refl.

Compute if (foo'' 3 4) then 1 else 0.
*)

Definition nat_eq: forall n m:nat, {n=m}+{n<>m}.
decide equality.
Defined.

Class EqDecSumbool (A : Type) := {
  eqb' : forall a1 a2: A, {a1=a2}+{a1<>a2} 
}.

Instance nat_eq_dec : EqDecSumbool nat := {
  eqb' := nat_eq
}.

Definition list_nat_eq: forall n m:(list nat), {n=m}+{n<>m}. 
decide equality. 
decide equality. 
Defined. 

Instance list_nat_eq_dec : EqDecSumbool (list nat) := {
  eqb' := list_nat_eq
}.

Definition option_nat_eq: forall n m: (option nat), {n=m}+{n<>m}.
decide equality.
decide equality.
Defined.

Instance option_nat_eq_dec : EqDecSumbool (option nat) := {
  eqb' := option_nat_eq
}.

Compute eqb' (Some 5) (Some (1+4)).
Compute eqb' (Some 5) (Some 0).
Compute eqb' (Some 5) (None).
Compute eqb' (None) (Some 0).
Compute eqb' (None) (None).


(* Binary Trees *)

Inductive btree  (A : Set) : Set :=
  | NLeaf
  | NNode : btree A -> A -> btree A -> btree A
.

Eval compute in NLeaf.

(*
            8
           / \
          6   7
         / \   \
        5   3   2
*)
Definition five_node := (NNode nat (NLeaf nat) 5 (NLeaf nat)).
Definition three_node := (NNode nat (NLeaf nat) 3 (NLeaf nat)).
Definition two_node := (NNode nat (NLeaf nat) 2 (NLeaf nat)).
Definition six_node := (NNode nat five_node 6 three_node).
Definition seven_node := (NNode nat (NLeaf nat) 6 two_node).
Definition eight_node := (NNode nat six_node 6 seven_node).

Check eight_node.
Print eight_node.

(*
forall P Q: Prop, Prop := P /\ Q, 

P /\ Q -- type, it has a proof if P has a proof and Q has a proof
*)

Inductive and (P Q: Prop): Prop :=
  conj: P -> Q -> and P Q
.

Inductive or (P Q: Prop): Prop :=
  | left: P -> (or P Q)
  | right: Q -> (or P Q)
.

Definition x: or (1=0) (2=3).
apply left.
Abort.

Definition not (P: Prop): Prop := P -> False.

Definition notOneIsZero: not (1=0).
compute.
intros.
inversion H.
Defined.

(*

forall P Q: Prop, and P Q -> P.

forall P: Prop, P -> P.

forall n: nat, n = n.

*)

Definition all (A: Set) (P: A -> Prop) := 
  forall a:A, P a
.

Definition neqn: all nat (fun a:nat => (a=a)).
compute.
intros.
trivial.
Defined.

Inductive ex (A: Set) (P: A -> Prop) := 
  ex_intro: forall a:A, P a -> ex A P
.

Definition existsNEq0: ex nat (fun n:nat => n=0).
apply ex_intro with (a:=0).
trivial.
Defined.

Definition id {A: Type} (a: A) : (A) := a.
Compute @id nat 2.
Compute id 2.

Class Functor (ftor: Set -> Set) := {
  fmap : forall {A B: Set}, (A -> B) -> (ftor A) -> (ftor B) ;
  ftor_id_law : forall A:Set, forall f:ftor A, fmap id f = f ;
  ftor_composition_law : 
      forall A B C:Set, 
      forall g: B->C, 
      forall f:A->B, 
      forall l:ftor A, 
      (fmap g (fmap f l)) = (fmap (fun a:A => g (f a)) l)
}.

Require Import List.

Instance list_ftor: Functor (list) := {
  fmap := map
}.
Proof.
(* id *) 
intros.
induction f.
trivial.
simpl.
rewrite -> IHf.
compute.
trivial.

(* comp *)
intros.
induction l.
simpl. trivial.
simpl. 
rewrite <- IHl.
trivial.
Qed.

Instance option_ftor: Functor (option) := {
  fmap := (fun {A B: Set} (func:A -> B) (input :option A) =>
              match input with 
                | None => None
                | Some a => Some (func a)
              end
            )
}.
Proof.
(* id *) 
intros.
destruct f.
compute. trivial.
trivial.
(* comp *)
intros.
destruct l.
simpl. trivial.
simpl. 
trivial.
Qed.

Import ListNotations.

Check fmap (S) [1].
Compute fmap (@id nat) [1;2;3].
Compute fmap (S) [1;2;3].
Compute fmap (S) (Some 1).
Compute fmap (fun n:nat => n*n) (Some 5).
Compute fmap (fun n:nat => n*n) None.
Compute fmap (fun n:nat => n*n) [1;2;3].

Definition arrow (R A: Set) := R->A.

Check arrow.

Check arrow nat. 
Check arrow bool. 

(* Definition arrow_map (R A B: Set) (f: A->B) (fa: arrow R A) : (arrow R B) := (fun (r: R) => f (fa r)) . *)
Definition arrow_map (R A B: Set) (f: A->B) (fa: (R->A) ) : (R->B) := (fun (r: R) => f (fa r)) .

Instance fun_ftor (R: Set) : Functor (arrow R) := {
  fmap := (arrow_map R)
}.
Proof.

(*id*)
intros.
compute.
trivial.

(* comp *)
intros.
compute.
trivial.
Qed.

Definition compose {A B C: Set} (f: A->B) (g: B->C) : (A->C) := fun (a:A) => (g (f a)).

Compute fmap S S 3.

(*
Class Applicative_Functor (f : Type -> Type) := {
  pure : forall a, a -> (f a) ;
  afmult : forall a b, f ( a -> b ) -> f a -> f b
}.

Instance applicative_functor_list : Applicative_Functor (list) := { 
  
  pure A := fun a:A => [a]; 
  
  afmult A B := (fix afmult_list (func_list: list (A->B)) (input_list: list A) := 
    match func_list with 
      | nil => nil
      | func_h :: func_t => (fmap A B func_h input_list) ++ (afmult_list func_t input_list)
    end )
}.

Check plus 1.
Check mult 0.
Compute afmult nat nat [plus 1; mult 10] [1;2;3].
*)

Class Applicative (F: Set -> Set) := {
  functor :> Functor F ; 
  ap: forall A B: Set, F (A -> B) -> F A -> F B ; 
  pure : forall a:Set, a -> (F a)
  (*
  afmult : forall (a b:Set), F ( a -> b ) -> F a -> F b
  *)
}.

Definition option_ap (A B: Set) (O_f: option (A->B)) (O_v: option A) : (option B) := 
match O_f with 
  | None => None
  | Some f => 
    match O_v with 
      | None => None
      | Some v => Some (f v) 
    end
end
.

Instance af_option : Applicative (option) := { 
  ap := option_ap ; 
  pure := Some
}.

Compute ap nat nat (pure (nat->nat) S) (Some 5).

Definition mult3 (a b c: nat) : nat := a*b*c.

(*

Compute ap
           (ap
               (ap 
               (pure mult3)
               (pure 3))
           (pure 4)
        (pure 5)

Compute ap
           (ap
               (ap 
               (@None (nat->nat->nat->nat))
               (pure 3))
           (pure 4)
        (pure 5)

Instance applicative_functor_list : Applicative (list) := { 
  
  pure A := fun a:A => [a]; 
  
  afmult A B:= (fix afmult_list (func_list: list (A->B)) (input_list: list A) := 
    match func_list with 
      | nil => nil
      | func_h :: func_t => (fmap func_h input_list) ++ (afmult_list func_t input_list)
    end )
}.

Check plus 1.
Check mult 0.
Compute afmult nat nat [plus 1; mult 10] [1;2;3].
Compute fmap_2 (fun a:nat => a+1) [1;2;3].
*)







