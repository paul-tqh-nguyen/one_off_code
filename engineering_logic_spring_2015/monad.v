
Require Omega.
Require Import Arith.
Require Export Arith.EqNat. 
Require Export Bool.
Require Export List.
Require Import Ascii.
Require String. Open Scope string_scope.
Require Import Coq.Strings.String.

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
end.

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
end.

Definition bgt_nat (a b : nat) : bool := negb (ble_nat a b).
Definition bge_nat (a b : nat) : bool := negb (blt_nat a b).

Fixpoint count (v:nat) (s:list nat) : nat := 
  match s with
    | nil => 0
    | h :: t => 
      match (beq_nat v h) with
        | true => 1 + (count v t)
        | _ => (count v t)
      end
  end
.

Definition member (v:nat) (s:list nat) : bool := blt_nat 0 (count v s).

Definition id {A: Type} (a: A) : (A) := a.
Compute @id nat 2.
Compute id 2.

Class Functor (ftor: Set -> Set) := {
  fmap : forall {A B: Set}, (A -> B) -> (ftor A) -> (ftor B) ;
  ftor_id_law : forall A:Set, forall f:ftor A, fmap id f = f ;
  ftor_composition_law : 
      forall A B C:Set, 
      forall g: B->C, 
      forall f: A->B, 
      forall l: ftor A, 
      (fmap g (fmap f l)) = (fmap (fun a:A => g (f a)) l)
}.

Require Import List.
Import ListNotations.

Definition option_fmap := 
            (fun {A B: Set} (func:A -> B) (input :option A) =>
              match input with 
                | None => None
                | Some a => Some (func a)
              end
            ).

Compute option_fmap (fun a:nat => a*a) (Some 9). 

Instance option_ftor: Functor (option) := {
  fmap := @option_fmap
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
Defined.

Compute fmap id (Some 5).
Compute fmap (fun n:nat => n*n) (Some 9).
Eval compute in fmap (fun n:nat => n*n) (Some 9).
Compute fmap (fun n:nat => n*n) None.

Fixpoint list_fmap {A B: Set} (func: A -> B) (input :list A) := 
match input with 
  | nil => nil
  | h :: t => (func h) :: (list_fmap func t)
end.

Instance list_ftor: Functor (list) := {
  fmap := @list_fmap
}.
Proof.
(* id *) 
intros.
induction f.
compute. trivial.
simpl. 
rewrite IHf. 
compute. trivial.
(* comp *)
intros.
induction l.
simpl. trivial.
simpl. rewrite IHl. 
trivial.
Defined.

Compute fmap (fun x => x*x) [1;2;3].

Definition compose {A B C: Type} (f: B->C) (g: A->B) : (A->C) := (fun (a:A) => f (g a) ).
Compute (compose S (fun n:nat => n*n)) 2.
Definition apply_fun {A B: Set} (a: A) (f: A -> B) := f a.
Compute (apply_fun 3) S.

Class Applicative_Functor (f : Set -> Set) := { 
  functor :> Functor f; 
  pure : forall {a : Set}, a -> (f a) ; 
  afmult : forall {A B : Set}, f ( A -> B ) -> f A -> f B ; 
  aftor_identity_law : forall A:Set, forall v: f A, afmult (pure (@id A)) v = v ; 
  aftor_composition_law : forall {a b c : Set} (u : f (b -> c)) (v : f (a -> b)) (w : f a),
         (afmult (afmult (afmult (pure compose) u) v) w)
         = afmult u (afmult v w) ; 
  aftor_homomorphism_law: forall {a b : Set} (f : a -> b) (x : a),
         afmult (pure f) (pure x) = pure (f x) ; 
  aftor_interchange_law: forall (A B: Set), forall y: A, forall u: f (A -> B), 
         afmult u (pure y) = afmult (pure (apply_fun y)) u 
}. 

Notation "f <*> x" := (afmult f x) (at level 42).

Definition option_afmult (A B: Set) (O_f: option (A->B)) (O_v: option A) : (option B) := 
match O_f with 
  | None => None
  | Some f => 
    match O_v with 
      | None => None
      | Some v => Some (f v) 
    end
end
.

Instance option_aftor: Applicative_Functor (option) := {
  pure := Some ; 
  afmult := option_afmult
}.
Proof.
(* aftor_identity_law *)
intros.
destruct v.
compute. trivial.
compute. trivial.
(* aftor_composition_law *)
intros. 
destruct u.
destruct v.
destruct w.
compute. trivial. 
compute. trivial.
compute. trivial.
compute. trivial.
(* aftor_homomorphism_law *)
intros.
compute. trivial.
(* aftor_interchange_law *)
intros.
compute. trivial.
Defined. 

Compute afmult (pure S) (Some 5).

Class Monoid (m : Set) := {
  mempty : m ; 
  mappend : m -> m -> m ; 
  (* mconcat : (list Set) -> m  ; *)
  (* mconcat = foldr mappend mempty *)  
  monoid_identity_law_left: forall x: m, 
      mappend mempty x = x ;  
  monoid_identity_law_right: forall x: m, 
      mappend x mempty = x ; 
  monoid_associativity_law: forall x y z: m, 
      mappend (mappend x y) z = mappend x (mappend y z)
}.

Fixpoint foldr {X Y:Type} (f: X->Y->Y) (b:Y) (l:list X) : Y :=
  match l with
  | nil => b
  | h :: t => f h (foldr f  b t) 
  end
.

Compute 5-6.
Compute foldr (fun (a b: nat) => a-b) 1 [6;7;3].

Fixpoint list_concat {A: Set}(l1 l2: list A) :=
  match l1 with 
    | nil => l2
    | h :: t => h :: (list_concat t l2)
  end.

Instance nat_list_monoid: Monoid (list nat) := { 
  mempty := [] ; 
  mappend := list_concat 
}. 
Proof. 
(* monoid_identity_law_left *)
intros.
compute. trivial.
(* monoid_identity_law_right *)
intros.
induction x.
compute. trivial.
simpl.
rewrite IHx.
trivial.
(* monoid_associativity_law *)
intros.
induction x.
induction y.
induction z.
compute. trivial.
compute. trivial.
compute. trivial.
simpl. rewrite IHx. trivial.
Defined.

Print string.

Class Monad (m : Type -> Type) := { 
  monad_return : forall {a: Type}, a -> m a ; 
  bind : forall {a b: Type}, m a -> (a -> m b) -> m b ; 
  double_arrow : forall {a b: Type}, m a -> m b -> m b (*; 
  monad_left_identity_law : forall {x y: Type} {a: x} {f: x -> (m y)}, (bind (monad_return a) f) = f a ; 
  monad_right_identity_law : forall {x: Type} {a: m x}, (bind a monad_return) = a ; 
  monad_associativity_law : forall {x y z: Type} {a: m x} {f: x -> (m y)} {g: y -> (m z)}, 
                              (bind (bind a f) g) = (bind a (fun (input:x) => (bind (f input) g)) ) 
  *)
}. 

Notation "m >>= f" := (bind m f) (at level 42).
Notation "m0 >> m1" := (double_arrow m0 m1) (at level 42).

Definition option_bind := 
(fun {a b: Type} (m_a: option a) (func: a -> (option b) ) => 
   match m_a with 
     | None => None
     | Some m_a' => (func m_a')
   end
).

Instance option_monad: Monad (option) := {
  monad_return := Some ; 
  bind := @option_bind;
  double_arrow := 
      (fun {a b: Type} x y (*(x: option a) (y: option b)*) => 
          option_bind x (fun (c:a) => y)
      ) 
}.
(*
Proof.
(* monad_left_identity_law *)
intros. compute. trivial. 
(* monad_right_identity_law *)
intros.
destruct a.
compute. trivial.
compute. trivial. 
(* monad_associativity_law *)
intros. 
destruct a.
simpl. trivial.
compute. trivial.
Defined.
*)

Compute bind (monad_return 4) (fun (x: nat) => Some (x*x)) = (fun (x: nat) => Some (x*x)) 4.
Compute fmap S [1;2;3].

(*
Definition list_bind {a b: Set} (m_a: list a) (func: a -> (list b) ) := 
  foldr list_concat [] (list_fmap func m_a).

Instance list_monad: Monad (list) := {
  monad_return := (fun {a: Set} (input: a) => cons input []) ; 
  bind := @list_bind;
  double_arrow := 
      (fun {a b: Set} x y (*(x: list a) (y: list b)*)  => 
          list_bind x (fun (c:a) => y)
      ) 
}.
Proof.
(* monad_left_identity_law *)
intros.
induction a.
compute.
trivial.
(* monad_right_identity_law *)
intros.
destruct a.
compute. trivial.
compute. trivial. 
(* monad_associativity_law *)
intros. 
destruct a.
simpl. trivial.
compute. trivial.
Defined.

Definition get_empty_nat_list (x: nat) : (list nat) := [].

Compute []:(list nat).
Compute bind [2;3] (fun (x: nat) => [x;x*x]).
Compute [2;3] >>= (fun (x: nat) => [x;x*x]).
Compute (list_fmap get_empty_nat_list [1;2;3]).
Compute [2;3] >>= get_empty_nat_list.
Compute [2;3] >>= (fun (x: nat) => [x;x*x]) >>= (fun (x: nat) => [x;x+1]) >>= (fun (x: nat) => ([]:(list nat))).
Compute [2;3] >> []:(list nat).
Compute [2;3] >>= (fun (x: nat) => [[x;10];[x;100]]).
Compute [3;4;5] >>= (fun (x: nat) => [x;2*x]).
*)

Record StateTrans (s a: Type) : Type :=
  ST {
    runState: s -> (s*a)
  }
.

Compute ST nat nat (fun (x:nat) => (x,x)).

Definition applyST := 
  (* applyST :: StateTrans s a -> s -> (s, a) *)
  fun {s a: Type} (state_transformer: StateTrans s a) (state_0: s) => 
    runState s a state_transformer state_0
.

Compute ST nat.
Check ST nat.
Check ST nat nat.
Check ST nat nat (fun x => (x,x)).
Check ST nat nat (fun x => (x,x)).
Check StateTrans.
Check StateTrans nat nat.
Compute StateTrans nat nat.
Compute runState nat nat (ST nat nat (fun x => (x,x))) 1.
Compute applyST (ST nat nat (fun x => (x,x))) 1.

Definition state_nat_monad_bind := 
              fun {a b: Type} (processor: StateTrans nat a) (processorGenerator: a -> StateTrans nat b) => 
                  ST nat b (fun state_0 => 
                             let (state_1, x) := applyST processor state_0
                             in applyST (processorGenerator x) state_1)
              .

Instance state_nat_monad: Monad (StateTrans nat) := {
  monad_return a := fun x:a => ST nat a (fun st => (st,x));
  bind a b := @state_nat_monad_bind a b;
  double_arrow a b := 
      (fun x y => 
          state_nat_monad_bind (x) (fun (c:a) => (y))
      )
}.
(*
Proof.
(* monad_left_identity_law *)
(* monad_right_identity_law *)
(* monad_associativity_law *)
Admitted.
*)

Definition ImpState := prod nat nat.
Check ImpState.
Compute ImpState.

Definition getX := ST ImpState nat ( 
                       fun (pair:ImpState) => 
                         match pair with 
                           | (x, y) => ((x,y),x)
                         end 
                     ).

Definition getY := ST ImpState nat ( 
                       fun (pair:ImpState) => 
                         match pair with 
                           | (x, y) => ((x,y),y)
                         end 
                     ).

Compute tt.

Definition putX (x':nat) := ST ImpState unit (
                               fun (pair:ImpState) => 
                                 match pair with 
                                   | (x, y) => ((x',y),tt)
                                 end 
                              ).

Definition putY (y':nat) := ST ImpState unit (
                               fun (pair:ImpState) => 
                                 match pair with 
                                   | (x, y) => ((x,y'),tt)
                                 end 
                              ).

Definition ImpState_monad_bind := 
              fun {a b: Type} (processor: StateTrans ImpState a) (processorGenerator: a -> StateTrans ImpState b) => 
                  ST ImpState b (fun state_0 => 
                             let (state_1, x) := applyST processor state_0
                             in applyST (processorGenerator x) state_1)
              .

Instance ImpState_monad: Monad (StateTrans ImpState) := {
  monad_return a := fun x:a => ST ImpState a (fun st => (st,x));
  bind a b := @ImpState_monad_bind a b;
  double_arrow a b := 
      (fun x y => 
          ImpState_monad_bind (x) (fun (c:a) => (y))
      )
}.

Compute applyST (putY 4 >>= (fun x => putX 9)) (222,333).

Definition getX_thenY :=
    getX >>= (fun x => 
    getY
    ).
Compute applyST getX_thenY (9,8).

Notation "*do* x <- f_0 ; f_1" := (f_0 >>= (fun x => f_1)) (at level 42).

Definition putSumIntoX := 
    *do* x <- getX ; 
    *do* y <- getY ; 
         putX (x+y).
Compute applyST putSumIntoX (2,3).

(*******************************************************)
(* Simple Example *)
Definition get3 := ST ImpState nat ( 
                       fun (pair:ImpState) => (pair,3)
                     ).
Definition get4 := ST ImpState nat ( 
                       fun (pair:ImpState) => (pair,4)
                     ).
Definition simpleExample := 
    *do* x <- get3 ;
    *do* y <- get4 ;
         monad_return (x+y).
Compute applyST simpleExample (2,1).
Definition simpleExample2 := 
    *do* x <- get4 ;
    *do* x <- get3 ;
         monad_return (x).
Compute applyST simpleExample2 (1,0).
(*******************************************************)

(* 
thing1 >>= \x ->
func1 x >>= \y ->
thing2 >>= \_ ->
func2 y >>= \z ->
return z

do
  x <- thing1
  y <- func1 x
  thing2
  z <- func2 y
  return z
*)

(*
Fixpoint gcdST : (StateTrans ImpState nat) := 
  getX >>= (fun x => 
  getY >>= (fun y => 
  (if beq_nat x y 
     then monad_return x
     else if ble_nat x y
            then 
              putY (y-x) >>= (fun _ =>
              gcdST)
            else 
              putY (x-y) >>= (fun _ =>
              gcdST)
  )
  )).
*)

Record StateTransEx (s a: Type) : Type :=
  STE {
    runStateEx: s -> option (s*a)
  }
.

Definition applySTE := 
  (* applySTE :: StateTransEx s a -> s -> option (s, a) *)
  fun {s a: Type} (state_transformer: StateTransEx s a) (state_0: s) => 
    runStateEx s a state_transformer state_0
.

Definition ImpStateEx_monad_bind := 
              fun {a b: Type} (processor: StateTransEx ImpState a) (processorGenerator: a -> StateTransEx ImpState b) => 
                  STE ImpState b (fun state_0 => 
                             match (applySTE processor state_0) with 
                               | Some (state_1, x) => applySTE (processorGenerator x) state_1 
                               | None => None 
                             end 
                             ) 
              . 

Instance ImpStateEx_monad: Monad (StateTransEx ImpState) := {
  monad_return a := fun x:a => STE ImpState a (fun st => Some (st,x));
  bind a b := @ImpStateEx_monad_bind a b;
  double_arrow a b := 
      (fun x y => 
          ImpStateEx_monad_bind (x) (fun (c:a) => (y))
      )
}.

Definition QState := ( (list nat)*(list nat)*(list nat) )%type.
Compute QState.

Definition getCols := STE QState (list nat) (
  fun (triple:QState) => 
    match triple with 
      | (cols, swDiags, seDiags) => Some ((cols, swDiags, seDiags),cols)
    end 
  ).
Compute getCols.
Compute applySTE getCols ([1], [2], [3]).

Definition getSWDiags := STE QState (list nat) (
  fun (triple:QState) => 
    match triple with 
      | (cols, swDiags, seDiags) => Some ((cols, swDiags, seDiags),swDiags)
    end 
  ).
Compute getSWDiags.
Compute applySTE getSWDiags ([1], [2], [3]).

Definition getSEDiags := STE QState (list nat) (
  fun (triple:QState) => 
    match triple with 
      | (cols, swDiags, seDiags) => Some ((cols, swDiags, seDiags),seDiags)
    end 
  ).
Compute getSEDiags.
Compute applySTE getSEDiags ([1], [2], [3]).

Definition putCols (c:nat) := STE QState (list nat) (
  fun (triple:QState) => 
    match triple with 
      | (cols, swDiags, seDiags) => Some ((c::cols, swDiags, seDiags), []:(list nat) )
    end 
  ).
Compute applySTE (putCols 99) ([1], [2], [3]).

Definition putSWDiags (sw:nat) := STE QState (list nat) (
  fun (triple:QState) => 
    match triple with 
      | (cols, swDiags, seDiags) => Some ((cols, sw::swDiags, seDiags), []:(list nat) )
    end 
  ).
Compute applySTE (putSWDiags 99) ([1], [2], [3]).

Definition putSEDiags (se:nat) := STE QState (list nat) (
  fun (triple:QState) => 
    match triple with 
      | (cols, swDiags, seDiags) => Some ((cols, swDiags, se::seDiags), []:(list nat) )
    end 
  ).
Compute applySTE (putSEDiags 99) ([1], [2], [3]).

Definition QStateEx_monad_bind := 
              fun {a b: Type} (processor: StateTransEx QState a) (processorGenerator: a -> StateTransEx QState b) => 
                  STE QState b (fun state_0 => 
                             match (applySTE processor state_0) with 
                               | Some (state_1, x) => applySTE (processorGenerator x) state_1 
                               | None => None 
                             end 
                             ) 
              .

Instance QStateEx_monad: Monad (StateTransEx QState) := {
  monad_return a := fun x:a => STE QState a (fun st => Some (st,x));
  bind a b := @QStateEx_monad_bind a b;
  double_arrow a b := 
      (fun x y => 
          QStateEx_monad_bind (x) (fun (c:a) => (y))
      )
}.

Definition tryPutCol (c:nat) := 
  getCols >>= (fun (cols:(list nat)) => 
  (if member c cols 
     then (STE QState (list nat) (fun s => (None:(option (QState * list nat))) )) (* mzero *) 
     else putCols c 
  ) 
). 
Compute applySTE (tryPutCol 99) ([1], [2], [3]).
Compute applySTE (tryPutCol 1) ([1], [2], [3]).

Definition tryPutSWDiag (sw:nat) := 
  getSWDiags >>= (fun (swDiags:(list nat)) => 
  (if member sw swDiags
     then (STE QState (list nat) (fun s => (None:(option (QState * list nat))) )) (* mzero *)
     else putSWDiags sw
  )
).
Compute applySTE (tryPutSWDiag 99) ([1], [2], [3]).
Compute applySTE (tryPutSWDiag 2) ([1], [2], [3]).

Definition tryPutSEDiag (se:nat) := 
  getSEDiags >>= (fun (seDiags:(list nat)) => 
  (if member se seDiags
     then (STE QState (list nat) (fun s => (None:(option (QState * list nat))) )) (* mzero *)
     else putSEDiags se
  )
).
Compute applySTE (tryPutSEDiag 99) ([1], [2], [3]).
Compute applySTE (tryPutSEDiag 3) ([1], [2], [3]).

Check tryPutSEDiag.

Definition StateTransEx_mplus {s a:Type} (p q: StateTransEx s a) := 
  STE s a (fun s0 => match (applySTE p s0) with
                       | Some (s1, a) => Some (s1, a)
                       | None => applySTE q s0
                     end)
.
Compute applySTE
        (@StateTransEx_mplus QState (list nat) 
            (putCols 99)
            (STE QState (list nat) (fun s => (None:(option (QState * list nat))) ))
        ) ([1], [2], [3]).
Compute applySTE
        (@StateTransEx_mplus QState (list nat) 
            (STE QState (list nat) (fun s => (None:(option (QState * list nat))) ))
            (putCols 99)
        ) ([1], [2], [3]).
Compute applySTE
        (@StateTransEx_mplus QState (list nat) 
            (putCols 88)
            (putCols 99)
        ) ([1], [2], [3]).
Compute applySTE
        (@StateTransEx_mplus QState (list nat) 
            (putCols 99)
            (putCols 88)
        ) ([1], [2], [3]).

Fixpoint tryEach {a b: Type} (values:list a) (f: a -> StateTransEx QState b) : (StateTransEx QState b) :=
  match values with 
    | nil => ( STE QState b (fun s => None) )
    | h::t => StateTransEx_mplus (f h) (tryEach t f)
  end 
.
Compute applySTE (tryEach [4;5;6] putCols) ([1], [2], [3]).
Compute applySTE (tryEach [1;1;6;4] tryPutCol) ([1], [2], [3]).

Fixpoint list_nat_comprehension (min max: nat) : list nat :=
  if beq_nat min max 
    then [min]
    else if bgt_nat min max 
           then []
           else 
             match max with 
               | S max' =>  (list_nat_comprehension min max')++[max]
               | _ => [] 
             end
.
Compute list_nat_comprehension 3 5.
Compute list_nat_comprehension 5 1.
Notation "x ... y" := (list_nat_comprehension x y) (at level 42).
Compute 1...5.
Compute 10...5.
Compute (0...(4-1)).

Definition place (r c offset:nat) := 
    *do* _ <- tryPutCol c ;
    *do* _ <- tryPutSWDiag (offset+c-r) ;
              tryPutSEDiag (c+r) 
.
Compute applySTE (place 1 2 0) ([], [], []). (* Expected Answer: (2,1,3) *)
Compute applySTE (place 2 1 0) ([], [], []). (* Expected Answer: (1,0,3) *)
Compute applySTE (place 2 1 2) ([], [], []). (* Expected Answer: (1,1,3) *) (* Answer w/o Offset: (1,-1,3) *)

Fixpoint queens (r colNum offset:nat) := 
  match r with 
    | O => getCols
    | S r' => tryEach 
                   (0...(colNum-1)) 
                   (fun c =>
                       *do* _ <- place (r') c offset;
                            queens (r') colNum offset)
  end
.

Definition n_queens_STE (n:nat) := (queens n n n).

Compute applySTE (n_queens_STE 8) ([], [], []).
Compute applySTE (n_queens_STE 4) ([], [], []).
Compute applySTE (n_queens_STE 1) ([], [], []).

Definition extractColsFromQstate (input:QState) : (list nat) := 
  match input with
    | (ans,_,_) => ans
  end
.

Definition extractSWDiagsFromQstate (input:QState) : (list nat) := 
  match input with
    | (_,ans,_) => ans
  end
.

Definition extractSEDiagsFromQstate (input:QState) : (list nat) := 
  match input with
    | (_,_,ans) => ans
  end
.

Fixpoint convert_to_coords_helper (cols seDiags: list nat) : list (nat*nat) :=
  match cols, seDiags with
    | hc::tc, hse::tse => (hc,hse-hc)::(convert_to_coords_helper tc tse)
    | _,_ => []
  end
.

Definition convert_to_coords (input:QState) : (list (nat*nat)) := 
  convert_to_coords_helper (extractColsFromQstate input) (extractSEDiagsFromQstate input)
.

Definition solve_n_queens (n:nat) : (list (nat*nat)) := 
  match ( applySTE (n_queens_STE n) ([], [], []) ) with
    | Some ((cols,swDiags,seDiags),outputs) => (convert_to_coords (cols,swDiags,seDiags) )
    | None => []
  end
.

Compute solve_n_queens 1.
Compute solve_n_queens 2.
Compute solve_n_queens 3.
Compute solve_n_queens 4.
Compute solve_n_queens 8.










































(********************************************************************************************************)

(* Nondeterministic N Queens Problem *)

Record StateTransMany (s a: Type) : Type :=
  STM {
    (* take one start state and returns all valid states that work if we start after that one *)
    runStateMany: s -> list (s*a) 
  }
.

Definition applySTM := 
  fun {s a: Type} (state_transformer: StateTransMany s a) (state_0: s) => 
    runStateMany s a state_transformer state_0
.

Fixpoint QStateMany_monad_bind_helper {s a b: Type} (input_list:list (s*a))
         (processorGenerator: a -> StateTransMany s b) : list (s*b) := 
  match input_list with
    | (state,ans)::t => (applySTM (processorGenerator ans) state)++(QStateMany_monad_bind_helper t processorGenerator)
    | [] => []
  end
.

Definition QStateMany_monad_bind := 
  fun {a b: Type} (processor: StateTransMany QState a) (processorGenerator: a -> StateTransMany QState b) => 
      STM QState b (fun state_0 => 
          QStateMany_monad_bind_helper (applySTM processor state_0) processorGenerator
      ). 

Instance QStateMany_monad: Monad (StateTransMany QState) := {
  monad_return a := fun x:a => STM QState a (fun st => [(st,x)] );
  bind a b := @QStateMany_monad_bind a b;
  double_arrow a b := 
      (fun x y => 
          QStateMany_monad_bind (x) (fun (c:a) => (y))
      )
}.

Definition getColsMany : (StateTransMany QState (list nat)) := STM QState (list nat) (
  fun (triple:QState) => 
    match triple with 
      | (cols, swDiags, seDiags) => [((cols, swDiags, seDiags),cols)]
    end 
  ).
Compute getColsMany.
Compute applySTM getColsMany ([1;1;1], [2;2], [3;3]).

Definition getSWDiagsMany : (StateTransMany QState (list nat)) := STM QState (list nat) (
  fun (triple:QState) => 
    match triple with 
      | (cols, swDiags, seDiags) => [((cols, swDiags, seDiags),swDiags)]
    end 
  ).
Compute getSWDiagsMany.
Compute applySTM getSWDiagsMany ([1;1;1], [2;2], [3;3]).

Definition getSEDiagsMany : (StateTransMany QState (list nat)) := STM QState (list nat) (
  fun (triple:QState) => 
    match triple with 
      | (cols, swDiags, seDiags) => [((cols, swDiags, seDiags),seDiags)]
    end 
  ).
Compute getSEDiagsMany.
Compute applySTM getSEDiagsMany ([1;1;1], [2;2], [3;3]).

Definition putColsMany (c:nat) := STM QState (list nat) (
  fun (triple:QState) => 
    match triple with 
      | (cols, swDiags, seDiags) => [((c::cols, swDiags, seDiags), []:(list nat) )]
    end 
  ).
Compute applySTM (putColsMany 99) ([1], [2], [3]).

Definition putSWDiagsMany (sw:nat) := STM QState (list nat) (
  fun (triple:QState) => 
    match triple with 
      | (cols, swDiags, seDiags) => [((cols, sw::swDiags, seDiags), []:(list nat) )]
    end 
  ).
Compute applySTM (putSWDiagsMany 99) ([1], [2], [3]).

Definition putSEDiagsMany (se:nat) := STM QState (list nat) (
  fun (triple:QState) => 
    match triple with 
      | (cols, swDiags, seDiags) => [((cols, swDiags, se::seDiags), []:(list nat) )]
    end 
  ).
Compute applySTM (putSEDiagsMany 99) ([1], [2], [3]).

Definition tryPutColMany (c:nat) := 
  getColsMany >>= (fun (cols:(list nat)) => 
  (if member c cols 
     then (STM QState (list nat) (fun s => ([]:(list (QState * list nat))) )) (* mzero *) 
     else putColsMany c 
  ) 
). 
Compute applySTM (tryPutColMany 99) ([1], [2], [3]).
Compute applySTM (tryPutColMany 1) ([1], [2], [3]).

Definition tryPutSWDiagMany (sw:nat) := 
  getSWDiagsMany >>= (fun (swDiags:(list nat)) => 
  (if member sw swDiags
     then (STM QState (list nat) (fun s => ([]:(list (QState * list nat))) )) (* mzero *)
     else putSWDiagsMany sw
  )
).
Compute applySTM (tryPutSWDiagMany 99) ([1], [2], [3]).
Compute applySTM (tryPutSWDiagMany 2) ([1], [2], [3]).

Definition tryPutSEDiagMany (se:nat) := 
  getSEDiagsMany >>= (fun (seDiags:(list nat)) => 
  (if member se seDiags
     then (STM QState (list nat) (fun s => ([]:(list (QState * list nat))) )) (* mzero *)
     else putSEDiagsMany se
  )
).
Compute applySTM (tryPutSEDiagMany 99) ([1], [2], [3]).
Compute applySTM (tryPutSEDiagMany 3) ([1], [2], [3]).

Definition StateTransMany_mplus {s a:Type} (p q: StateTransMany s a) : (StateTransMany s a) := 
  STM s a (fun s0 => (applySTM p s0)++(applySTM q s0) )
.
Compute applySTM
        (@StateTransMany_mplus QState (list nat) 
            (putColsMany 99)
            (STM QState (list nat) (fun s => ([]:(list (QState * list nat))) ))
        ) ([1], [2], [3]).
Compute applySTM
        (@StateTransMany_mplus QState (list nat) 
            (STM QState (list nat) (fun s => ([]:(list (QState * list nat))) ))
            (putColsMany 99)
        ) ([1], [2], [3]).
Compute applySTM
        (@StateTransMany_mplus QState (list nat) 
            (putColsMany 88)
            (putColsMany 99)
        ) ([1], [2], [3]).
Compute applySTM
        (@StateTransMany_mplus QState (list nat) 
            (putColsMany 99)
            (putColsMany 88)
        ) ([1], [2], [3]).
Compute applySTM
        (@StateTransMany_mplus QState (list nat) 
            (tryPutColMany 99)
            (tryPutColMany 88)
        ) ([1], [2], [3]).
Compute applySTM
        (@StateTransMany_mplus QState (list nat) 
            (tryPutColMany 1)
            (tryPutColMany 88)
        ) ([1], [2], [3]).
Compute applySTM
        (@StateTransMany_mplus QState (list nat) 
            (tryPutColMany 88)
            (tryPutColMany 1)
        ) ([1], [2], [3]).
Compute applySTM
        (@StateTransMany_mplus QState (list nat) 
            (tryPutSEDiagMany 3)
            (tryPutColMany 1)
        ) ([1], [2], [3]).
Compute applySTM
        (@StateTransMany_mplus QState (list nat) 
            (tryPutSEDiagMany 99)
            (tryPutColMany 88)
        ) ([1], [2], [3]).

Fixpoint tryEachMany {a b: Type} (values:list a) (f: a -> StateTransMany QState b) : (StateTransMany QState b) :=
  match values with 
    | nil => ( STM QState b (fun s => []) )
    | h::t => StateTransMany_mplus (f h) (tryEachMany t f)
  end 
.
Compute applySTM (tryEachMany [4;5;6] tryPutColMany) ([1], [2], [3]).
Compute applySTM (tryEachMany [1;1;6;4] tryPutColMany) ([1], [2], [3]).

Definition placeMany (r c offset:nat) := 
    *do* _ <- tryPutColMany c ;
    *do* _ <- tryPutSWDiagMany (offset+c-r) ;
              tryPutSEDiagMany (c+r) 
.
Compute applySTM (placeMany 1 2 0) ([], [], []). (* Expected Answer: (2,1,3) *)
Compute applySTM (placeMany 2 1 0) ([], [], []). (* Expected Answer: (1,0,3) *)
Compute applySTM (placeMany 2 1 2) ([], [], []). (* Expected Answer: (1,1,3) *) (* Answer w/o Offset: (1,-1,3) *)

Fixpoint queensMany (r colNum offset:nat) := 
  match r with 
    | O => getColsMany
    | S r' => tryEachMany 
                   (0...(colNum-1)) 
                   (fun c =>
                       *do* _ <- placeMany (r') c offset;
                            queensMany (r') colNum offset)
  end
.
Compute applySTM (queensMany 4 4 4) ([], [], []).

Definition extractQstateFromStateAnsPair (input:(QState * list nat)%type) := 
  match input with
    | ((cols,swDiags,seDiags),outputs) => (cols,swDiags,seDiags)
  end
.
Compute fmap (fun x => convert_to_coords (extractQstateFromStateAnsPair x)) (applySTM (queensMany 4 4 4) ([], [], [])).

Definition n_queens_STM (n:nat) := (queensMany n n n).

Definition solve_n_queens_many (n:nat) : (list (list (nat*nat))) := 
  fmap (fun x => convert_to_coords (extractQstateFromStateAnsPair x)) ( applySTM (n_queens_STM n) ([], [], []) )
.
Compute solve_n_queens_many 4.
Compute solve_n_queens_many 8.
Compute length (solve_n_queens_many 8).

















