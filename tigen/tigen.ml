(*
 * Graduate Programming Languages - Wes Weimer
 *
 * Test Input Generation Project 
 *
 * Summary: Given a C file, produce a suite of test inputs for that file
 *        such that branch coverage is maximized. 
 *
 * Input: A filename for a pre-processed, instrumented C file. This file
 *        is typically obtained via the "make test" Makefile target or
 *        the "instrument" utility. 
 *        example: test/simple-1/simple-1.i
 *
 * Output: Create a series of C files test0001.c, test0002.c, etc., 
 *        in that same directory. Each has main() defined to call
 *        the last function defined in the input file with the 
 *        goal of visiting every labeled statement. 
 *
 * High-Level Approach: 
 *  Step 1 -- generate straight-line paths of statements and "assumes"
 *  Step 2 -- symbolically execute those paths  
 *  Step 3 -- generate constraints from symex states
 *  Step 4 -- solve those constraints (obtaining values for variables)
 *  Step 5 -- convert constraint solutions into test inputs
 *
 * Many places in this code are annotated with "Possible FIXME", indicating
 * an area that might be expanded in a student project. 
 *)
open Utils
open Cil

module Z3Int = Z3.Arithmetic.Integer
module Z3Real = Z3.Arithmetic.Real

let do_debug = ref false

(* Paul's shortcuts *)
let large_int = 1 lsl 20
let car = List.hd
let cdr = List.tl
let clamp (x:int64) (minimum:int64) (maximum:int64) = min maximum (max minimum x)
let print = Printf.printf
let print_stmt (input : Cil.stmt) : unit = 
  let cil_printer_instance = new Cil.defaultCilPrinterClass in 
    Cil.dumpStmt cil_printer_instance stdout 4 input ; Printf.printf "\n"
let print_expr (input : Cil.exp) : unit = 
  let cil_printer_instance = new Cil.defaultCilPrinterClass in 
    Printf.printf "%s\n" (Pretty.sprint 1000 (Cil.printExp cil_printer_instance () input) )
let print_typ (input : Cil.typ) : unit = 
    let output_string = 
        match input with
            | TVoid(_) -> "TVoid(_)" 
            | TInt(_,_) -> "TInt(_,_)" 
            | TFloat(_,_) -> "TFloat(_,_)" 
            | TPtr(_,_) -> "TPtr(_,_)" 
            | TArray(_,_,_) -> "TArray(_,_,_)" 
            | TFun(_,_,_,_) -> "TFun(_,_,_,_)" 
            | TNamed(_,_) -> "TNamed(_,_)" 
            | TComp(_,_) -> "TComp(_,_)" 
            | TEnum(_,_) -> "TEnum(_,_)"
            | TBuiltin_va_list(_) -> "TBuiltin_va_list(_)"
            | _ -> "MISC"
    in
    print_string (output_string ^ "\n")
let get_case_expr_from_label (input : Cil.label) : Cil.exp =  
    begin
        match input with 
            | Case(exp, loc) -> exp
            | Default(loc) -> Const(CStr("DEFAULT!"))
    end

(**********************************************************************
 * Path Enumeration
 *
 * In this step we take as input a C function and statically enumerate a
 * set of paths through that function. A path is a list of executed
 * statements (e.g., "x=2;" along the path) intermixed with assumptions
 * (i.e., if the path corresponds to the true branch in "if (x < 5) {...}"
 * then you can assume "x < 5" after that point on the path). 
    (* Paul Note: Shouldn't we use "e.g." instead of "i.e." immediately above? *)
 *
 * Because a function may have many paths, we use a worklist to keep track
 * of which parts we are currently exploring. 
 *)

type path_exploration = 
  | Exploring_Block of Cil.block 
        (* a Cil.block is just a set of attributes (whatever those are) and a List of Cil.stmts
         * http://www.cs.berkeley.edu/~necula/cil/api/Cil.html#TYPEblock*)
  | Exploring_Statement of Cil.stmt  
  | Exploring_Done 

type path_step =
  | Statement of Cil.stmt 
  | Assume of Cil.exp 

type path = path_step list 

(* Paul's Shortcut *)
let print_path_exploration (x:path_exploration) : unit = 
    begin
        match x with 
            | Exploring_Block(b) -> Printf.printf "\nBlock:\n" ; List.iter print_stmt b.bstmts ; Printf.printf "\n" 
            | Exploring_Statement(s) -> Printf.printf "\nStatement:\n" ; print_stmt s ; Printf.printf "\n" 
            | Exploring_Done -> Printf.printf "\nDone\n" ; Printf.printf "Done\n"
    end
let print_path_step (x:path_step) : unit = 
    begin
        match x with 
            | Statement(s) -> print_stmt s 
            | Assume(e) -> print_expr e 
    end 
let print_path (x:path) : unit = 
    List.iter print_path_step x

let path_enumeration
  (target_fundec : Cil.fundec) (* method to enumerate paths in *) 
  : (path list) (* outputs : paths through that method *) 
  = 
  let enumerated_paths = ref [] in (* gather up our final answer *) 
  let note_path (p : path) = enumerated_paths := p :: !enumerated_paths in 
    (* This functions just makes sure to add the paths specified to the list of enumerated paths *)
  
  (*
   * Each worklist element will contain a five-tuple: 
   * (1) the visited path so far,
   * (2) the current place to explore
   * (3) where to go if the current exploration terminates normally
   * (4) where to go if the current exploration is "break;" 
   * (5) where to go if the current exploration is "continue;" 
   *)
  let worklist = Queue.create () in 
    (* queue of statements in the code we need to explore *)
  
  let add_to_worklist path where nn nb nc = 
    (* Possible FIXME: To avoid infinite loops in our analysis, if
     * we would enqueue a visit to a statement we've already visited _in
     * this path's history_, we instead give up immediately. *) 
    match where with
    | Exploring_Statement(s) when 
      List.exists (fun already_visited -> match already_visited with
        | Statement(visited_s) when visited_s.sid = s.sid -> true
        | _ -> false
        (* this func returns true if any path_step in our path matches the current path_step we're considering adding, so we don't add it  *)
      ) path -> Queue.add (path,Exploring_Done,[],[],[]) worklist 
    | _ -> Queue.add (path,where,nn,nb,nc) worklist 
  in 
  
  (* We start enumerating at the first line of the function body. *) 
  add_to_worklist [] (Exploring_Block(target_fundec.sbody)) [] [] [] ;
            (* target_fundec.sbody returns a Cil.block of Stmts *)
            (* this adds all the lines of the function we're analyzing to our worklist of paths to explore *)
  
  while not (Queue.is_empty worklist) do
    (* nn = next normal
     * nb = next if we hit a "break;"
     * nc = next if we hit a "continue;" *)
    let path, here, nn, nb, nc = Queue.pop worklist in 
    let give_up () = 
      (* At various times we will stop exploring along a path but we'll
       * still want to report that path. This function handles such cases. *) 
      add_to_worklist (path) (Exploring_Done) [] [] []
    in 
    (* The heart of path enumeration is a giant switch statement on 
     * the structure of the code being explored. *) 
    match here with

    | Exploring_Done -> begin 
        match nn with
        | [] -> note_path path (* we're done with this path! *) 
        | first :: rest -> 
          (* We might be done exploring the inside of a "then-branch",
           * for example, but we should then fall through and explore
           * whatever came after that whole if. *) 
          add_to_worklist path first rest nb nc 
      end 

    | Exploring_Block(b) -> begin
        match b.bstmts with
          | [] -> add_to_worklist path (Exploring_Done) nn nb nc 
          | first :: rest -> 
            (* if we hit a block with statements "S1; S2; S3;", 
             * we'll schedule a visit to S1 right away and put
             * "S2; S3;" on the list of things to visit next. *) 
            let followup = (Exploring_Block { b with bstmts = rest }) in 
            add_to_worklist path (Exploring_Statement(first)) 
              (followup :: nn) nb nc 
      end 

    | Exploring_Statement(s) -> begin
      match s.skind with

      | Instr _ -> (* e.g., handle "x = 2;" *) 
        add_to_worklist (Statement(s) :: path) (Exploring_Done) nn nb nc

      | Return(exp_option,loc) -> 
        (* Possible FIXME: This is not (yet) an interprocedural analysis. *)
            ( 
                begin 
                    match exp_option with 
                        | None -> add_to_worklist (path) (Exploring_Done) nn nb nc 
                        | Some v -> add_to_worklist (path) (Exploring_Done) nn nb nc 
                end 
            ) ; 

      | Goto(goto_target,_) -> 
        (* Possible FIXME: Handle totally unstructured programs. *) 
(*        add_to_worklist (path) (Exploring_Block { battrs = []; bstmts = (!goto_target :: (!goto_target).succs) }) nn nb nc ;*)
        add_to_worklist (path) (Exploring_Statement(!goto_target)) (List.map (fun x -> Exploring_Statement(x) ) (!goto_target).succs) nb nc  
(*        add_to_worklist (path) (Exploring_Statement(!goto_target)) nn nb nc  *)
(*        add_to_worklist (path) (Exploring_Done) [Exploring_Statement(!goto_target)] [] []  *)
      
      | Switch(expression, block, statement_list,location) -> 
        (* Possible FIXME: Handle switch statements. *) 
(*        List.iter print_stmt statement_list ; *)
        List.iter (fun (x:Cil.stmt) -> 
                        ( 
                            begin 
                                match car x.labels with 
                                    | Case(case_exp, loc) -> 
                                        let assumption = Assume (BinOp(Eq,expression,case_exp, Cil.intType)) in 
                                        add_to_worklist (assumption :: path) (Exploring_Block(block)) nn nb nc 
                                    | Default(loc) -> 
                                        add_to_worklist (path) (Exploring_Block(block)) nn nb nc 
                            end 
                        ) 
                  ) statement_list 
      | TryFinally _ (* Microsoft C Extension *) 
      | TryExcept _ (* Microsoft C Extension *) 
      -> give_up () 
      
      | Break _ -> begin
          match nb, nc with 
          | b_hd :: b_tl , c_hd :: c_tl -> 
            add_to_worklist path (Exploring_Done) b_hd b_tl c_tl 
          | _, _ -> 
            (* break with no enclosing loop structure *)
            give_up () 
        end 
      
      | Continue _ -> begin 
          match nb, nc with 
          | b_hd :: b_tl , c_hd :: c_tl -> 
            add_to_worklist path (Exploring_Done) c_hd b_tl c_tl 
          | _, _ -> 
            (* continue with no enclosing loop structure *) 
            give_up () 
        end 
      
      | If(exp,then_branch,else_branch,_) -> 
        (* As usual in Axiomatic Semantics, when exploring the Then-Branch 
         * you get to assume the conditional is True, and when exploring
         * the Else-Branch you get to assume that it is false. *) 
        let then_condition = exp in
        let else_condition = UnOp(LNot,exp,(Cil.typeOf exp)) in (* == !exp *)
        add_to_worklist  ((Assume then_condition) :: path) 
          (Exploring_Block(then_branch)) nn nb nc ;
        add_to_worklist  ((Assume else_condition) :: path) 
          (Exploring_Block(else_branch)) nn nb nc 

      | Loop(loop_block,_,break_opt,continue_opt) -> 
        (* In CIL, while (b) { c } becomes
         *
         * while (1) {
         *   if (!b) break; 
         *   c;
         * } 
         *
         * Thus all Loops are the equivalent of "while true". *)  
        add_to_worklist path (Exploring_Block loop_block) 
          (here :: nn) 
          (nn :: nb) 
          ((here :: nn) :: nc) 

      | Block(b) -> 
        add_to_worklist path (Exploring_Block b) nn nb nc 

    end 
  done ;

  (* We prepended statements to the front of paths, so we have to
   * reverse them to get the right history order. *) 
  let paths = List.map List.rev !enumerated_paths in 

  debug "tigen: %s: %d path(s) enumerated\n" 
    target_fundec.svar.vname 
    (List.length paths) ;
  
(*  List.iter (fun x -> print "--------------------------------------\n" ; print_path x) paths;*)
  
  paths 

(**********************************************************************
 * Symbolic Variable State (or Symbolic Register File) 
 *
 * Our state is a simple mapping from variable names to symbolic
 * expressions. We use the existing Cil.exp expression type for
 * symbolic expressions as well.
 *)
module OrderedString =
  struct
    type t = string
    let compare = compare
  end
module StringMap = Map.Make(OrderedString)
module StringSet = Set.Make(OrderedString)

type symbolic_variable_state = Cil.exp StringMap.t 

let empty_symbolic_variable_state = StringMap.empty 

(* The usual state update: sigma[variable_name := new_value] *) 
let symbolic_variable_state_update 
  (sigma : symbolic_variable_state)  
  (variable_name : string)
  (new_value : Cil.exp) 
  : symbolic_variable_state
  =
  StringMap.add variable_name new_value sigma 

(*
 * Look up a variable in the symbolic state. For example, if we know that
 * [x=10] and [y=z+3] and we lookup "y", we expect to get "z+3" out.
 *)
let symbolic_variable_state_lookup 
      (sigma : symbolic_variable_state) 
      (variable : Cil.exp) 
      : Cil.exp =
  let found = match variable with
  | Lval(Var(va),NoOffset) -> 
    begin
      try
        Some(StringMap.find va.vname sigma)
      with Not_found -> 
        None
    end 
  | Lval(Mem(exp),NoOffset) -> None (* cannot handle memory access *) 
  | Lval(lhost,Field(_)) -> None (* cannot handle field access *) 
  | Lval(lhost,Index(_)) -> None (* cannot handle array index *) 
  | _ -> None (* not a variable *) 
  in 
  
  match found with
  | Some(answer) -> answer
  | None -> variable 

(*
 * Rewrite an expression based on the current symbolic state.  For example,
 * if we know that [x=10] and [y=z+3] and we lookup "sin(x+y)", we expect
 * to get "sin(10+z+3)". 
 *
 * We use Cil's visitor pattern to implement this.
 * http://en.wikipedia.org/wiki/Visitor_pattern
 *)
  class substituteVisitor (sigma : symbolic_variable_state) = object
    inherit nopCilVisitor
    method vexpr e = 
      ChangeDoChildrenPost(e,(fun e ->
        symbolic_variable_state_lookup sigma e
      ))
  end 

let symbolic_variable_state_substitute 
      (sigma : symbolic_variable_state) 
      (exp : Cil.exp) 
      : Cil.exp =
  let sv = new substituteVisitor sigma in 
  visitCilExpr sv exp 

(**********************************************************************
 * Symbolic Execution
 *
 * We build on the "symbolic register file" code above to implement a more
 * generic symbolic execution. Given a "path" (a sequence of statements and
 * assumptions) we update our symbolic register file when we encounter
 * assignment statements and then record every assumption as we make it. 
 *
 * Later, we'll feed those assumptions as constraints to an automated
 * theorem prover to generate test inputs. 
 *)

type symex_state = {
  register_file : symbolic_variable_state ;
  assumptions : Cil.exp list ;
} 

let empty_symex_state = {
  register_file = empty_symbolic_variable_state ;
  assumptions = [] ; 
} 

(*  class noteVarVisitor (varset : StringSet.t ref) = object*)
  class noteVarVisitor (varset : Cil.varinfo list ref) = object
    inherit nopCilVisitor
    method vvrbl v = 
      varset := List.append [v] !varset ;
      DoChildren
  end 

(* Given a path, produce a final symbolic execution state (a symbolic
 * register file and set of assumptions) associated with the end of that
 * path. *) 
let symbolic_execution
  (path : path) 
  : symex_state 
  = 

  if false then begin (* enable this for symex debugging *) 
    (* Paul: This just prints every path_step currently in path *)
    debug "\ntigen: symex:\n" ;
    List.iter (fun step -> 
      match step with
      | Statement(s) -> 
        debug "%s\n" (Pretty.sprint ~width:80 (dn_stmt () s)) 
      | Assume(e) -> 
        debug "Assume %s\n" (Pretty.sprint ~width:80 (dn_exp () e)) 
    ) path ;
  end ;

  let state = empty_symex_state in 
  (* For each variable mentioned in the path, assign it a default,
   * arbitrary value. We use "_x" to represent the unknown initial
   * value of variable "x". 
   *
   * Possible FIXME: This may not handle memory (i.e., arrays, pointers)
   * correctly. *) 
(*  let variables = ref StringSet.empty in *)
  let variables = ref [] in 
  let nv = new noteVarVisitor variables in 
  List.iter (fun step -> match step with
    | Statement(s) -> ignore (visitCilStmt nv s) 
    | Assume(e) -> ignore (visitCilExpr nv e) 
  ) path ; 
  let new_register_file = (* StringSet.fold *) 
(*    List.fold_right (fun variable_name state ->*)
    List.fold_right (fun var_info state ->
    let new_value = Lval(Var(makeVarinfo false ("_" ^ var_info.vname) 
      (var_info.vtype)),NoOffset) in 
    symbolic_variable_state_update state var_info.vname new_value 
  ) !variables state.register_file in 
  let state = { state with register_file = new_register_file } in 

  (*
   * Now we walk down every step in the path, handling assignment
   * statements (which update the symbolic register file) and assumptions
   * (which are evaluated and gathered up). 
   *)
  let final_state = List.fold_left (fun state step ->
    match step with
    | Assume(e) -> (* recall that we get these from "if" statements. *)
      let evaluated_e = symbolic_variable_state_substitute 
        state.register_file e in
      { state with assumptions = evaluated_e :: state.assumptions} 
    | Statement(s) -> begin
      match s.skind with
      | Instr(il) -> 
        List.fold_left (fun state instr ->
          match instr with
          | Set((Var(va),NoOffset),rhs,_) -> 
            let evaluated_rhs = symbolic_variable_state_substitute 
              state.register_file rhs 
            in 
            let new_register_file = symbolic_variable_state_update 
              state.register_file va.vname evaluated_rhs in
            { state with register_file = new_register_file } 
          | Set((Mem(address),_),rhs,_) ->
            (* Possible FIXME: cannot handle memory accesses like *p *) state 
          | Set((_,Field(f,o)),rhs,_) -> 
            (* Possible FIXME: cannot handle field accesses like e.fld *) state 
          | Set((_,Index(i,o)),rhs,_) -> 
                (* Possible FIXME: cannot handle array indexing like a[i] *) 
                let evaluated_rhs = symbolic_variable_state_substitute 
                  state.register_file rhs 
                in 
(*                let new_register_file = symbolic_variable_state_update *)
(*                  state.register_file va.vname evaluated_rhs in*)
(*                { state with register_file = new_register_file }*)
                state 
          | Call (lval_option, main_expression, expression_list, location)  -> 
            (* Possible FIXME: cannot handle function calls *)
            state
                
          | Asm _ -> (* cannot handle inline ASM *) state 
        ) state il 
      | _ -> state 
    end
  ) state path in

  final_state 

(**********************************************************************
 * Constraint Solving
 *
 * Given the final symbolic excution state corresponding to a path,
 * we now want to generate constraints for a theorem prover and solve those
 * constraints. For example, if we know that "x > 10" && "x < 15", we'd
 * like to come up with a concrete assignment like "x == 11". That concrete
 * value is a test input that forces execution down the path in question!
 *)

(* The final constraint solution will be a mapping from variable names to 
 * textual values (i.e., from "x" to "11"). Possible FIXME: This is
 * unlikely to be sufficient for more complicated values (e.g., pointers,
 * arrays).  *)
type solved_constraints = string StringMap.t 

let solve_constraints
  (target_fundec : Cil.fundec) (* method to generate inputs for *)
  (state : symex_state) (* final symex state associated with a path *)
  : (solved_constraints) option  (* Some x == path is feasible 
                                  * None   == path is NOT feasible *)
  = 
  (* We use the Z3 automated theorem prover and SMT solver. We need
   * more than a "yes or no" answer: we need a satisfying assignment (also
   * called a "model"). So we tell Z3 that we want such a model. *) 
  let ctx = Z3.mk_context [( "model", "true" )] in 
  let debug =
    if !do_debug then (* enable this for Z3 debugging *) 
      Utils.debug
    else
      fun fmt -> Printf.kprintf ignore fmt
  in 
  (* Much of the work here is converting from CIL Abstract Syntax Trees to
   * Z3 Abstract Syntax Trees. *) 
(*  let zero_ast_real = Z3Int.mk_numeral_i ctx 0 in *)
  let zero_ast = Z3Int.mk_numeral_i ctx 0 in 
  let undefined_ast = zero_ast in 

  (* Every time we encounter the same C variable "foo" we want to map
   * it to the same Z3 node. We use a hash table to track this. *) 
  let symbol_ht = Hashtbl.create 255 in
  let var_to_ast (var_name, variable_info) = (* goes from varinfo to ast *)
    try
      Hashtbl.find symbol_ht (var_name, variable_info)
    with _ -> 
      (* Possible FIXME: currently we assume all variables are integers. *)
      begin
        match variable_info.vtype with 
            | TFloat(_,_) -> 
                let ast = Z3Real.mk_const_s ctx var_name in
                Hashtbl.replace symbol_ht (var_name, variable_info) ast ;
                ast
(*            | TInt(IChar,_) -> *)
            | _ -> 
                let ast = Z3Int.mk_const_s ctx var_name in
                Hashtbl.replace symbol_ht (var_name, variable_info) ast ;
                ast
      end
  in 
  (* In Z3, boolean-valued and integer-valued expressions are different
   * (i.e., have different _Sort_s). CIL does not have this issue. *) 
  let is_binop exp = 
    match exp with 
    | UnOp(LNot,_,_) 
    | BinOp(Lt,_,_,_) 
    | BinOp(Le,_,_,_) 
    | BinOp(Gt,_,_,_) 
    | BinOp(Ge,_,_,_) 
    | BinOp(Eq,_,_,_) 
    | BinOp(Ne,_,_,_) -> true
    | _ -> false
  in 

  (* This is the heart of constraint generation. For every CIL expression
   * (e.g., "x > 10"), convert it to an equivalent Z3 expression. *) 
  let constraints = ref [] in
  let rec exp_to_ast (exp : Cil.exp) : Z3.Expr.expr =
    match exp with
    | Const(CInt64(i,_,_)) -> 
      (* Possible FIXME: large numbers are not handled *) 
      let i = Int64.to_int i in 
      Z3Int.mk_numeral_i ctx i

    | Const(CChr(c)) -> 
      (* Possible FIXME: characters are justed treated as integers *) 
      let i = Char.code c in
      Z3Int.mk_numeral_i ctx i
      
    | Const(CReal(float_value,f_kind,string_option)) -> 
        (* Paul: I added some stuff to deal with reals *)
        Z3Real.mk_numeral_nd ctx (int_of_float ((float_of_int large_int)*.float_value)) large_int
        
    | Const(_) -> 
      (* Possible FIXME: reals, enums, strings, etc., are not handled *) 
      undefined_ast

    | Lval(Var(va),NoOffset) -> var_to_ast (va.vname, va) 

    | Lval(_) -> 
      (* Possible FIXME: var.field, *p, a[i], etc., are not handled *) 
      undefined_ast

    | UnOp(Neg,e,_) -> Z3.Arithmetic.mk_unary_minus ctx (exp_to_ast e) 
    | UnOp(LNot,e,_) when is_binop e -> Z3.Boolean.mk_not ctx (exp_to_ast e) 
    | UnOp(LNot,e,_) -> Z3.Boolean.mk_eq ctx (exp_to_ast e) (zero_ast) 

    | BinOp(PlusA,e1,e2,_)  -> Z3.Arithmetic.mk_add ctx [ exp_to_ast e1; exp_to_ast e2]
    | BinOp(MinusA,e1,e2,_) -> Z3.Arithmetic.mk_sub ctx [ exp_to_ast e1; exp_to_ast e2]
    | BinOp(Mult,e1,e2,_)   -> Z3.Arithmetic.mk_mul ctx [ exp_to_ast e1; exp_to_ast e2]
    | BinOp(Div,e1,e2,_) -> 
      let ast2 = exp_to_ast e2 in 
      let not_div_by_zero = Z3.Boolean.mk_distinct ctx [ zero_ast ; ast2 ] in
      constraints := not_div_by_zero :: !constraints;
      Z3.Arithmetic.mk_div ctx (exp_to_ast e1) ast2 
    | BinOp(Mod,e1,e2,_) -> Z3Int.mk_mod ctx (exp_to_ast e1) (exp_to_ast e2) 
    | BinOp(Lt,e1,e2,_) -> Z3.Arithmetic.mk_lt ctx (exp_to_ast e1) (exp_to_ast e2) 
    | BinOp(Le,e1,e2,_) -> Z3.Arithmetic.mk_le ctx (exp_to_ast e1) (exp_to_ast e2) 
    | BinOp(Gt,e1,e2,_) -> Z3.Arithmetic.mk_gt ctx (exp_to_ast e1) (exp_to_ast e2) 
    | BinOp(Ge,e1,e2,_) -> Z3.Arithmetic.mk_ge ctx (exp_to_ast e1) (exp_to_ast e2) 
    | BinOp(Eq,e1,e2,_) -> Z3.Boolean.mk_eq ctx (exp_to_ast e1) (exp_to_ast e2) 
    | BinOp(Ne,e1,e2,_) -> 
      Z3.Boolean.mk_distinct ctx [ (exp_to_ast e1) ; (exp_to_ast e2) ] 
    | CastE(type_to_be_cast_to,e) -> 
            (* Possible FIXME: (int)(3.1415) ? *) 
            begin
                match type_to_be_cast_to with 
                    | TInt(IChar,_) -> (* cast to char *)
                        begin
                            match e with
                                | Const(CInt64(i,_,_)) -> (* if we're casting from int to char  *)
                                    let char_code = (clamp 
                                                        i 
                                                        (Int64.of_int 0) 
                                                        (Int64.of_int 255) 
                                                    ) in
                                    exp_to_ast ( Const(CChr( (Char.chr (Int64.to_int char_code) ) )) )
                                | Const(CChr(c)) -> (* if we're casting from char to char, i.e. doing nothing *)
                                    exp_to_ast e
                                | Const(CReal(float_value,f_kind,string_option)) -> (* if we're casting from float to char *)
                                    let char_code = (clamp 
                                                        (Int64.of_int (int_of_float float_value)) 
                                                        (Int64.of_int 0) 
                                                        (Int64.of_int 255) 
                                                    ) in
                                    let char_code64 = (Int64.to_int char_code) in
                                    let char_value = (Char.chr char_code64) in
                                    exp_to_ast (Const(CChr(char_value)))
                                | _ -> exp_to_ast e
                        end
                    | TInt(_,_) -> (* cast to int *)
                        begin
                            match e with
                                | Const(CInt64(i,_,_)) -> (* if we're casting from int to int, i.e. doing nothing  *)
                                    exp_to_ast e
                                | Const(CChr(c)) -> (* if we're casting from char to int *)
                                    let i = Char.code c in
                                    let i64 = (Int64.of_int i) in
                                    exp_to_ast (Const(CInt64(i64,IInt, Some (string_of_int i) )))
                                | Const(CReal(float_value,f_kind,string_option)) -> (* if we're casting from float to int *)
                                    let i = (int_of_float float_value) in
                                    let int64_value = Int64.of_int i in
                                    exp_to_ast (Const(CInt64(int64_value,IInt, Some (string_of_int i) )))
                                | _ -> exp_to_ast e
                        end
                    | TFloat(_,_) -> (* cast to real *)
                        begin
                            match e with
                                | Const(CInt64(i,_,_)) -> (* if we're casting from int to float  *)
                                    let float_value = float_of_int (Int64.to_int i) in
                                    exp_to_ast (Const(CReal(float_value,FDouble,Some (string_of_float float_value) )))
                                | Const(CChr(c)) -> (* if we're casting from char to float *)
                                    let i = Char.code c in
                                    let float_value = float_of_int (i) in
                                    exp_to_ast (Const(CReal(float_value,FDouble,Some (string_of_float float_value) )))
                                | Const(CReal(float_value,f_kind,string_option)) -> (* if we're casting from float to float, i.e. doing nothing *)
                                    exp_to_ast e
                                | _ -> exp_to_ast e
                        end
                    | _ -> exp_to_ast e
            end
    | _ -> 
      (* addrof, startof, alignof, sizeof, etc., are not handled *) 
      undefined_ast
  in 

  (* For every assumption along the path, convert it to a Z3 expression
   * and tell the theorem prover to assert it as true (i.e., as a
   * constraint). *) 
  List.iter (fun cil_exp -> 
    let cil_exp =
      if is_binop cil_exp then
        cil_exp
      else
        UnOp(LNot, UnOp(LNot, cil_exp, typeOf cil_exp), typeOf cil_exp)
    in
    try 
      debug "converting: %s\n" (Pretty.sprint ~width:80 (dn_exp () cil_exp));
      let z3_ast = exp_to_ast cil_exp in 
      (*
      debug "tigen: asserting %s\n" 
        (Z3.ast_to_string ctx z3_ast) ; 
      *) 
      constraints := z3_ast :: !constraints
    with _ -> begin  
    (*
      debug "tigen: cannot convert %s to Z3\n"
        (Pretty.sprint ~width:80 (dn_exp () cil_exp)) ;
        *) 
        ()
    end 
  ) state.assumptions ; 
  List.iter (fun cnstr ->
    debug "constraints: %s\n" (Z3.Expr.to_string cnstr)
  ) !constraints;

  (* Now that we've put in all of the constraints, query the theorem
   * prover to see if there is a model that can satisfy them all at the
   * same time. *) 
  let solver = Z3.Solver.mk_solver ctx None in
  Z3.Solver.add solver !constraints;
  let result =
    if (Z3.Solver.check solver []) = Z3.Solver.SATISFIABLE then
      (* If there is a model, we try to extra concrete values from it. Those
       * concrete values become our solution. *) 
      let Some(model) = Z3.Solver.get_model solver in
      let solution =
        List.fold_left (fun solution formal_variable ->
          let underscore_name = "_" ^ formal_variable.vname in 
          let z3_ast = var_to_ast (underscore_name,formal_variable) in 
          try begin
            match Z3.Model.get_const_interp_e model z3_ast with
            | Some(evaluated) ->
              let evaluated = Z3.Expr.to_string evaluated in
              if evaluated <> "" && evaluated.[0] <> '_' then
                StringMap.add formal_variable.vname evaluated solution
              else
                solution
            | None ->
              solution
          end with Z3.Error msg ->
            solution
        ) StringMap.empty target_fundec.sformals
      in
      Some(solution)
    else
      None
  in
  result 

(**********************************************************************
 * Emit Test Case
 * 
 * Given a concrete solution (e.g., "x = 5", "y = 22"), we must
 * actually emit a test case that calls the method in question with
 * those parameters. For this project, we emit every test case as a
 * separate C file so that we can compile them all separately and
 * calculate the total coverage dynamically. 
 *)
let emit_test_case
  (target_fundec : Cil.fundec) (* method to generate inputs for *) 
  (filename : string) (* where to put this test *) 
  (solution : solved_constraints) (* what values to use *) 
  : unit (* outputs results to disk *) 
  = 

  let fout = open_out filename in 
  let extern_decl = GVarDecl(target_fundec.svar,locUnknown) in 
  (* First, if we're a test case for gcd(int, int), add an
   * forward declaration to tell the compiler that function gcd(int, int)
   * exists. *)
  Printf.fprintf fout "#include <stdio.h>\n\nextern %s\n\n" 
    (Pretty.sprint ~width:80 (dn_global () extern_decl)) ; 

  (* We emit our test case as a little main() program. *) 
  Printf.fprintf fout "int main() {\n" ; 

  (* Declare local variables to hold all of the formals. *) 
  List.iter (fun formal ->
    Printf.fprintf fout "\t%s %s;\n" 
      (Pretty.sprint ~width:80 (dn_type () formal.vtype))
      formal.vname ; 
  ) target_fundec.sformals ; 

  (* The subject program may loop forever, but we don't want to. Break it
   * off after a few seconds. *) 
  Printf.fprintf fout "\talarm(2);\n" ; 

  (* If our solution says that "x" maps to "2", add "x = 2;" to the test
   * case. *) 
  List.iter (fun formal -> 
    try 
      let value = StringMap.find formal.vname solution in 
            begin
                match formal.vtype with
                    | TFloat(float_kind,_) -> (* Floats and Doubles*)
                        let suffix = 
                            begin
                                match float_kind with 
                                    | FFloat -> "f"
                                    | FDouble -> ""
                                    | _ -> ""
                            end
                        in 
                        if String.contains value '('
                            then   
                                  let two_nums_string_sub = (String.sub value 3 ((String.length value)-4)) in 
                                  let space_index = String.index two_nums_string_sub ' ' in 
                                  let numerator_string = 
                                        (String.sub two_nums_string_sub 
                                                    0 
                                                    space_index)
                                        in 
                                  let denominator_string = 
                                        (String.sub two_nums_string_sub 
                                                    (succ space_index)
                                                    ((String.length two_nums_string_sub)-space_index-1) )
                                        in 
                                  let numerator_float = float_of_string numerator_string in 
                                  let denominator_float = float_of_string denominator_string in 
                                  let final_value_float = numerator_float /. denominator_float in
                                  let final_out_string = (string_of_float final_value_float) ^ suffix
                                        in 
                                  Printf.fprintf fout "\t%s = %s;\n" formal.vname ( "(" ^ final_out_string ^ ")" )

                            else
                                Printf.fprintf fout "\t%s = %s;\n" 
                                    formal.vname value 
                    | TInt(IChar,_) -> (* chars, which are kind of like ints *)
                        Printf.fprintf fout "\t%s = %s;\n" 
                            formal.vname ("'" ^ (Char.escaped (Char.chr (int_of_string value))) ^ "'")
                    | _ -> (* since we assume everything else is an int *)
                        Printf.fprintf fout "\t%s = %s;\n" 
                            formal.vname value 
            end
    with _ -> () 
  ) target_fundec.sformals ; 

  (* Now all that's left to do is actually call the function. That will
   * look something like: "gcd(a,b);" *)
  let var_to_exp v = Lval(Var(v),NoOffset) in 
  let actuals = List.map var_to_exp target_fundec.sformals in
  let call_instr = Call(None,
    var_to_exp target_fundec.svar, actuals, locUnknown) in 

  Printf.fprintf fout "\n\n\t%s\n\treturn 0;\n}\n" 
    (Pretty.sprint ~width:80 (dn_instr () call_instr)) ; 

  close_out fout ; 
  () 

(**********************************************************************
 * Test Input Generation
 *
 * Generate test cases for the given function and write them to the given
 * directory. This is a direct implementation of the multi-step algorithm
 * in the top-level comment: enumerate paths, symbolically execute them, 
 * generate and solve constraints corresponding to those symex states,
 * and emit test cases based on those constraint solutions. 
 *)
let test_input_generation 
  (target_fundec : Cil.fundec) (* method to generate inputs for *) 
  (directory : string) (* where to put the tests *) 
  : unit (* outputs results to disk *) 
  = 

  let paths = path_enumeration target_fundec in 
      (* Paul: We'll magically call this function to create a list of paths. 
       * paths are of type List of path_step
       * path_steps are either statements Cil.Stmt or assumtions/assertions Cil.exp *)

  let paths = first_nth paths 500 in (* don't take too long! *) 

  let symbolic_states = List.map symbolic_execution paths in 

  (* We'll use a hashtbl as a cheap way to gather up unique
   * solutions since I can't be bothered to define a StringMapSet. *)
  let solutions = Hashtbl.create 255 in 
  List.iter (fun state ->
    match solve_constraints target_fundec state with
    | None -> () 
    | Some(answer) -> Hashtbl.replace solutions answer true 
  ) symbolic_states;

  let test_case_counter = ref 0 in (* how many tests generated so far? *) 
  let next_test_case_name () = 
    let local_name = Printf.sprintf "test%04d.c" !test_case_counter in
    incr test_case_counter ; 
    Filename.concat directory local_name
  in 

  Hashtbl.iter (fun solution _->
    emit_test_case 
      target_fundec
      (next_test_case_name ())
      solution 
  ) solutions ; 

  debug "tigen: %s: %d test case(s) emitted\n" 
    target_fundec.svar.vname 
    !test_case_counter ; 

  () 

(**********************************************************************
 * Main Driver
 *
 * We accept the program to test as the only command-line argument. We
 * emit the test cases in the same directory as the program source. Try
 * "make test" and "make eval" to run this automatically on the provided
 * tests. 
 *)
let main () = begin
  let usage   = "Usage: " ^ Sys.argv.(0) ^ " [options] filename" in
  let options =
    [ "--debug", Arg.Set(do_debug), " enable Z3 debugging messages"; ]
  in
  let options = Arg.align options in
  let args = ref [] in
  Arg.parse options (fun x -> args := x :: !args) usage;
  let filename =
    match List.rev !args with
    | [filename] -> filename
    | _ ->
      debug "tigen: specify a pre-preprocessed C file\n" ;
      Arg.usage options usage;
      exit 1 
  in

  Z3.toggle_warning_messages true ; 

  let directory = Filename.dirname filename in 
  let file = load_c_file filename in 
  (* 
   * We want each statement to have a unique statement ID, so we'll call
   * computeFileCFG. 
   *) 
  Cfg.computeFileCFG file ; (* Paul's not entirely sure what this funciton does *)

  (* Find the last fundec in the file. *) 
  let rec find_fundec global_list = 
    match global_list with
    | GFun(fd,loc) :: tl -> test_input_generation fd directory 
        (* GFun is  a type that represents global functions. We're just grabbing the function (the last one declared) and passing it to test_input_generation 
         * directory tells us where to put our results (same folder adn the input file)
         * http://www.cs.berkeley.edu/~necula/cil/api/Cil.html#TYPEglobal *)
    | hd :: tl -> find_fundec tl 
    | [] -> debug "tigen: no functions declared in %s\n" filename 
  in
  find_fundec (List.rev file.globals) ;

end ;;
main () ;;

(* tigen.ml: end of file *) 
