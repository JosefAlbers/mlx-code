# Copyright 2026 J Joe

from __future__ import annotations
_Q='decorator'
_P='end_lineno'
_O='lineno'
_N='replace'
_M='utf-8'
_L=False
_K='annotation'
_J='function'
_I='method'
_H='lines'
_G='file'
_F='kind'
_E='store'
_D='class'
_C='var'
_B='load'
_A=None
import ast,json,sys
from collections import defaultdict
from dataclasses import dataclass,field
from pathlib import Path
from typing import Optional
DEFAULT_BIND_DEPTH=4
@dataclass
class Definition:
	name:str;kind:str;module:str;file:str;lines:tuple[int,int];cls:Optional[str]=_A
	def qname(A):
		if A.cls:return f"{A.module}.{A.cls}.{A.name}"
		return f"{A.module}.{A.name}"
	def to_dict(A):return{'qname':A.qname(),_F:A.kind,_G:A.file,_H:list(A.lines)}
	def display(A):B=f"{A.file}:{A.lines[0]}-{A.lines[1]}";return f"  DEF  {A.kind:8s}  {A.qname()}  [{B}]"
@dataclass
class Use:
	name:str;kind:str;file:str;lines:tuple[int,int];scope:str;snippet:str;via:Optional[str]=_A
	def to_dict(A):
		B={'name':A.name,_F:A.kind,_G:A.file,_H:list(A.lines),'scope':A.scope,'snippet':A.snippet}
		if A.via is not _A:B['via']=A.via
		return B
	def display(A):B=f"  (via {A.via})"if A.via else'';C=f"{A.lines[0]}-{A.lines[1]}"if A.lines[1]!=A.lines[0]else str(A.lines[0]);return f"  {A.kind:14s}  {A.file}:{C:<9s}  [{A.scope}]  {A.snippet}{B}"
@dataclass
class Results:
	query:str|list[str];definitions:list[Definition]=field(default_factory=list);uses:list[Use]=field(default_factory=list);seed_names:set[str]=field(default_factory=set);alias_map:dict[str,str]=field(default_factory=dict)
	def to_dict(A):B={B:C for(B,C)in A.alias_map.items()if B not in A.seed_names};return{'query':A.query,'seeds':sorted(A.seed_names),'aliases':B,'definitions':[A.to_dict()for A in A.definitions],'uses':[A.to_dict()for A in A.uses]}
@dataclass
class Binding:lhs:str;rhs:str
class Indexer(ast.NodeVisitor):
	def __init__(A,module,file):B=module;A.module=B;A.file=file;A.definitions=[];A.bindings=[];A._class_stack=[];A._scope_stack=[B];A._func_depth=0
	@property
	def _scope(self):return self._scope_stack[-1]
	@property
	def _current_class(self):return self._class_stack[-1]if self._class_stack else _A
	def visit_ClassDef(A,node):B=node;C=B.lineno,B.end_lineno or B.lineno;A.definitions.append(Definition(name=B.name,kind=_D,module=A.module,file=A.file,lines=C));A._class_stack.append(B.name);A._scope_stack.append(f"{A.module}.{B.name}");A.generic_visit(B);A._scope_stack.pop();A._class_stack.pop()
	def visit_FunctionDef(A,node):
		B=node;E=B.lineno,B.end_lineno or B.lineno;F=_I if A._current_class else _J;A.definitions.append(Definition(name=B.name,kind=F,module=A.module,file=A.file,lines=E,cls=A._current_class));G=f"{A.module}.{A._current_class}.{B.name}"if A._current_class else f"{A.module}.{B.name}"
		for C in B.args.args+B.args.posonlyargs+B.args.kwonlyargs:
			if C.annotation:
				D=_bare_of(C.annotation)
				if D and C.arg!=D:A._add(C.arg,D)
		A._scope_stack.append(G);A._func_depth+=1;A.generic_visit(B);A._func_depth-=1;A._scope_stack.pop()
	visit_AsyncFunctionDef=visit_FunctionDef
	def visit_Assign(A,node):
		B=node
		if A._func_depth==0 and not A._current_class:
			for C in B.targets:
				if isinstance(C,ast.Name):
					D=B.lineno,B.end_lineno or B.lineno
					if not any(B.name==C.id and B.module==A.module for B in A.definitions):A.definitions.append(Definition(name=C.id,kind=_C,module=A.module,file=A.file,lines=D))
		for C in B.targets:A._record(C,B.value)
		A.generic_visit(B)
	def visit_AnnAssign(B,node):
		A=node
		if B._func_depth==0 and not B._current_class:
			if isinstance(A.target,ast.Name):
				C=A.lineno,A.end_lineno or A.lineno
				if not any(C.name==A.target.id and C.module==B.module for C in B.definitions):B.definitions.append(Definition(name=A.target.id,kind=_C,module=B.module,file=B.file,lines=C))
		if A.value:B._record(A.target,A.value)
		B.generic_visit(A)
	def visit_AugAssign(B,node):A=node;B._record(A.target,A.value);B.generic_visit(A)
	def visit_NamedExpr(B,node):A=node;B._record(A.target,A.value);B.generic_visit(A)
	def _add(A,a,b):
		if a and b and a!=b:A.bindings.append(Binding(a,b));A.bindings.append(Binding(b,a))
	def _record(B,lhs,rhs):
		A=_bare_of(lhs)
		if not A:return
		C=_rhs_name(rhs)
		if C:B._add(A,C)
		B._record_container(A,rhs)
	def _record_container(B,container_name,node):
		D=node;C=container_name
		if isinstance(D,ast.Dict):
			for E in D.values:
				if E is _A:continue
				A=_rhs_name(E)
				if A:B._add(C,A)
				else:B._record_container(C,E)
		elif isinstance(D,(ast.List,ast.Tuple,ast.Set)):
			for F in D.elts:
				A=_rhs_name(F)
				if A:B._add(C,A)
				else:B._record_container(C,F)
def expand_seeds(seeds,bindings,depth=DEFAULT_BIND_DEPTH):
	F=seeds;C=defaultdict(set)
	for A in bindings:C[A.lhs].add(A.rhs);C[A.rhs].add(A.lhs)
	B={A:A for A in F};G=set(F)
	for I in range(depth):
		D=set()
		for H in G:
			for E in C.get(H,set()):
				if E not in B:B[E]=B[H];D.add(E)
		if not D:break
		G=D
	return B
class UseScanner(ast.NodeVisitor):
	def __init__(A,module,file,source_lines,target_names,via_map):B=module;A.module=B;A.file=file;A.src=source_lines;A.target_names=target_names;A.via_map=via_map;A.uses=[];A._scope_stack=[B];A._class_stack=[];A._stmt_lines=0,0
	@property
	def _scope(self):return self._scope_stack[-1]
	@property
	def _current_class(self):return self._class_stack[-1]if self._class_stack else _A
	def _snip(A,line):return A.src[line-1].strip()if 1<=line<=len(A.src)else''
	def _via(B,name):A=B.via_map.get(name);return A if A and A!=name else _A
	def _hit(A,name,kind,node):
		D,E=A._stmt_lines
		if D:B=D,E
		else:C=getattr(node,_O,0);B=C,getattr(node,_P,C)or C
		A.uses.append(Use(name=name,kind=kind,file=A.file,lines=B,scope=A._scope,snippet=A._snip(B[0]),via=A._via(name)))
	def _match(A,name):return name in A.target_names
	def _stmt_ctx(A,node):return _StmtCtx(A,node)
	def visit_ClassDef(A,node):
		B=node
		with A._stmt_ctx(B):
			for C in B.decorator_list:A._check_expr(C,_Q)
			for D in B.bases:A._check_expr(D,_B)
		A._class_stack.append(B.name);A._scope_stack.append(f"{A.module}.{B.name}"if A.module else B.name)
		for E in B.body:A.visit(E)
		A._scope_stack.pop();A._class_stack.pop()
	def visit_FunctionDef(A,node):
		B=node
		with A._stmt_ctx(B):
			for E in B.decorator_list:A._check_expr(E,_Q)
			if B.returns:A._check_expr(B.returns,_K)
			for C in B.args.args+B.args.posonlyargs+B.args.kwonlyargs:
				if C.annotation:A._check_expr(C.annotation,_K)
			for D in B.args.defaults+B.args.kw_defaults:
				if D:A.visit(D)
		F=f"{A.module}.{A._current_class}.{B.name}"if A._current_class else f"{A.module}.{B.name}";A._scope_stack.append(F)
		for G in B.body:A.visit(G)
		A._scope_stack.pop()
	visit_AsyncFunctionDef=visit_FunctionDef
	def visit_Assign(A,node):
		C=node;G=A._scope==A.module and not A._current_class;D=_A
		if G:
			for B in C.targets:
				if isinstance(B,ast.Name):D=B.id;break
		F=_L
		if D:A._scope_stack.append(f"{A.module}.{D}");F=True
		with A._stmt_ctx(C):
			for B in C.targets:
				E=_bare_of(B)
				if E and A._match(E):A._hit(E,_E,B)
				else:A.visit(B)
			A._visit_rhs(C.value)
		if F:A._scope_stack.pop()
	def visit_AnnAssign(A,node):
		B=node;F=A._scope==A.module and not A._current_class;C=_A
		if F and isinstance(B.target,ast.Name):C=B.target.id
		E=_L
		if C:A._scope_stack.append(f"{A.module}.{C}");E=True
		with A._stmt_ctx(B):
			if B.annotation:A._check_expr(B.annotation,_K)
			D=_bare_of(B.target)
			if D and A._match(D):A._hit(D,_E,B.target)
			else:A.visit(B.target)
			if B.value:A._visit_rhs(B.value)
		if E:A._scope_stack.pop()
	def visit_AugAssign(A,node):
		B=node
		with A._stmt_ctx(B):
			C=_bare_of(B.target)
			if C and A._match(C):A._hit(C,_E,B.target)
			D=_bare_of(B.value)
			if D and A._match(D):A._hit(D,_B,B.value)
			else:A.visit(B.value)
	def visit_Return(B,node):
		A=node
		with B._stmt_ctx(A):
			if A.value:
				C=_bare_of(A.value)
				if C and B._match(C):B._hit(C,'return',A.value)
				else:B.visit(A.value)
	def visit_Expr(A,node):
		with A._stmt_ctx(node):A.visit(node.value)
	def _visit_generic_stmt(A,node):
		with A._stmt_ctx(node):A.generic_visit(node)
	visit_If=_visit_generic_stmt;visit_For=_visit_generic_stmt;visit_While=_visit_generic_stmt;visit_With=_visit_generic_stmt;visit_Try=_visit_generic_stmt;visit_Raise=_visit_generic_stmt;visit_Assert=_visit_generic_stmt;visit_Delete=_visit_generic_stmt;visit_Global=_visit_generic_stmt;visit_Nonlocal=_visit_generic_stmt
	def _check_expr(A,node,kind):
		B=node;C=_bare_of(B)
		if C and A._match(C):A._hit(C,kind,B)
		else:A.visit(B)
	def _visit_rhs(B,node):
		A=node
		if isinstance(A,ast.Dict):
			for D in A.values:
				if D is not _A:B._visit_rhs(D)
		elif isinstance(A,(ast.List,ast.Tuple,ast.Set)):
			for E in A.elts:B._visit_rhs(E)
		else:
			C=_bare_of(A)
			if C and B._match(C):B._hit(C,_B,A)
			else:B.visit(A)
	def visit_Call(A,node):
		I='arg';B=node
		if isinstance(B.func,ast.Subscript):
			E=_bare_of(B.func.value)
			if E and A._match(E):
				A._hit(E,'subscript_call',B)
				for C in B.args:A.visit(C)
				for D in B.keywords:A.visit(D.value)
				return
		F=_bare_of(B.func)
		if F and A._match(F):A._hit(F,'call',B)
		else:A.visit(B.func)
		for C in B.args:
			G=_bare_of(C)
			if G and A._match(G):A._hit(G,I,C)
			else:A.visit(C)
		for D in B.keywords:
			H=_bare_of(D.value)
			if H and A._match(H):A._hit(H,I,D.value)
			else:A.visit(D.value)
	def visit_Attribute(A,node):
		D='attr';B=node;C=_bare_of(B.value)
		if C and A._match(C):A._hit(C,D,B)
		if A._match(B.attr):A._hit(B.attr,D,B)
		if not(C and A._match(C)):A.visit(B.value)
	def visit_Subscript(A,node):
		B=node;C=_bare_of(B.value)
		if C and A._match(C):A._hit(C,_B,B)
		else:A.visit(B.value)
		A.visit(B.slice)
	def visit_Name(B,node):
		A=node
		if B._match(A.id):B._hit(A.id,_B,A)
	def visit_NamedExpr(B,node):
		A=node;C=_bare_of(A.value)
		if C and B._match(C):B._hit(C,_B,A.value)
		else:B.visit(A.value)
		if B._match(A.target.id):B._hit(A.target.id,_E,A.target)
class _StmtCtx:
	__slots__='scanner','prev'
	def __init__(C,scanner,node):A=scanner;C.scanner=A;B=getattr(node,_O,0);D=getattr(node,_P,B)or B;C.prev=A._stmt_lines;A._stmt_lines=B,D
	def __enter__(A):return A
	def __exit__(A,*B):A.scanner._stmt_lines=A.prev
def _expr_to_dotted(node):
	A=node
	if isinstance(A,ast.Name):return A.id
	if isinstance(A,ast.Attribute):
		B=_expr_to_dotted(A.value)
		if B:return f"{B}.{A.attr}"
def _bare_of(node):A=_expr_to_dotted(node);return A.split('.')[-1]if A else _A
def _rhs_name(node):
	A=node
	if isinstance(A,ast.Call):return _bare_of(A.func)
	return _bare_of(A)
def _module_name(root,file):
	try:A=file.with_suffix('').relative_to(root);return'.'.join(A.parts)
	except ValueError:return file.stem
def _collect_files(target):
	A=Path(target)
	if A.is_file():return A.parent,[A]
	return A,sorted(A for A in A.rglob('*.py')if'__pycache__'not in A.parts)
def _parse(f):
	try:A=f.read_text(encoding=_M,errors=_N);return ast.parse(A,filename=str(f)),A.splitlines()
	except SyntaxError:return
def analyze(target,query,depth=DEFAULT_BIND_DEPTH):return analyze_multi(target,[query],depth=depth)
def analyze_multi(target,queries,depth=DEFAULT_BIND_DEPTH):
	B=queries;V,W=_collect_files(target);A=Results(query=B if len(B)!=1 else B[0]);G=[]
	for H in W:
		N=_parse(H)
		if N is _A:continue
		C,E=N;G.append((C,E,_module_name(V,H),str(H)))
	O=[];P=[]
	for(C,E,I,J)in G:K=Indexer(I,J);K.visit(C);O.extend(K.definitions);P.extend(K.bindings)
	F=set()
	for Q in B:
		X=Q.split('.')[-1];F.add(X)
		for L in Q.split('.'):
			if len(L)>1 and L.isidentifier():F.add(L)
	A.definitions=[A for A in O if any(A.name==B.split('.')[-1]or A.name==B or A.qname()==B for B in B)];A.seed_names=F;M=expand_seeds(F,P,depth=depth);A.alias_map=M;Y=set(M.keys())
	for(C,E,I,J)in G:R=UseScanner(I,J,E,Y,M);R.visit(C);A.uses.extend(R.uses)
	S=set();T=[]
	for D in A.uses:
		U=D.file,D.lines,D.name,D.kind
		if U not in S:S.add(U);T.append(D)
	A.uses=sorted(T,key=lambda u:(u.file,u.lines[0],u.lines[1]));return A
@dataclass
class SymbolNode:
	name:str;kind:str;lines:tuple[int,int];children:list['SymbolNode']=field(default_factory=list)
	def line_count(A):return A.lines[1]-A.lines[0]+1
class OutlineVisitor(ast.NodeVisitor):
	def __init__(A,depth=1):A._max_depth=depth;A._stack=[];A.roots=[]
	def _current_depth(A):return len(A._stack)
	def _add(A,node):
		if A._stack:A._stack[-1].children.append(node)
		else:A.roots.append(node)
	def visit_ClassDef(A,node):
		B=node;D=B.lineno,B.end_lineno or B.lineno;C=SymbolNode(name=B.name,kind=_D,lines=D);A._add(C)
		if A._current_depth()<A._max_depth-1:A._stack.append(C);A.generic_visit(B);A._stack.pop()
	def _visit_func(A,node):
		B=node;D=B.lineno,B.end_lineno or B.lineno;E=_I if A._stack and A._stack[-1].kind==_D else _J;C=SymbolNode(name=B.name,kind=E,lines=D);A._add(C)
		if A._current_depth()<A._max_depth-1:A._stack.append(C);A.generic_visit(B);A._stack.pop()
	visit_FunctionDef=_visit_func;visit_AsyncFunctionDef=_visit_func
	def visit_Assign(B,node):
		A=node
		if B._stack:return
		for C in A.targets:
			if isinstance(C,ast.Name):D=A.lineno,A.end_lineno or A.lineno;B._add(SymbolNode(name=C.id,kind=_C,lines=D))
	def visit_AnnAssign(B,node):
		A=node
		if B._stack:return
		if isinstance(A.target,ast.Name):C=A.lineno,A.end_lineno or A.lineno;B._add(SymbolNode(name=A.target.id,kind=_C,lines=C))
def _outline_file_trees(target,depth):
	F,G=_collect_files(target);C=[]
	for A in G:
		D=_parse(A)
		if D is _A:continue
		H,I=D;B=OutlineVisitor(depth=depth);B.visit(H)
		if B.roots:
			try:E=str(A.relative_to(F))
			except ValueError:E=str(A)
			C.append((E,B.roots))
	return C
def _node_to_dict(n):
	A={'name':n.name,_F:n.kind,_H:list(n.lines)}
	if n.children:A['children']=[_node_to_dict(A)for A in n.children]
	return A
def format_outline(target,depth=1):
	D=_outline_file_trees(target,depth);E={_D:'C',_J:'f',_I:'m',_C:'v'}
	def B(n,indent):
		A=indent;D='  '*A;F=E.get(n.kind,'?');G=f"{n.lines[0]}-{n.lines[1]}"if n.line_count()>1 else str(n.lines[0]);C=[f"{D}[{F}] {n.name}  ({G})"]
		for H in n.children:C.extend(B(H,A+1))
		return C
	A=[]
	for(C,F)in D:
		A.append(f"\n{C}");A.append('─'*min(60,len(C)+4))
		for G in F:A.extend(B(G,indent=0))
	A.append('');return'\n'.join(A)
def outline_data(target,depth=1):return[{_G:A,'symbols':[_node_to_dict(A)for A in B]}for(A,B)in _outline_file_trees(target,depth)]
def _parse_args(argv):
	K='--no-expand';J='--nodefs';I='--raw';H='--json';E='--depth';C=argv;Q={H,I,J,K,E}
	if len(C)<1:print(__doc__,file=sys.stderr);sys.exit(1)
	L=C[0];A=C[1:];F=[];B=0
	while B<len(A)and not A[B].startswith('--'):F.append(A[B]);B+=1
	M=H in A;N=I in A;O=J not in A;G=K in A;D=DEFAULT_BIND_DEPTH
	if G:D=0
	if E in A:
		P=A.index(E)
		try:D=int(A[P+1])
		except(IndexError,ValueError):print('--depth requires an integer argument',file=sys.stderr);sys.exit(1)
	return L,F,M,N,O,G,D
CONTEXT_LINES=3
def format_results(r,show_defs=True,raw=_L):
	V=',  ';N=', ';E=raw;A=[];W=N.join(r.query)if isinstance(r.query,list)else r.query
	if not E:
		A.append(f"\nQuery  : {W!r}");A.append(f"Seeds  : {N.join(sorted(r.seed_names))}");O={A:B for(A,B)in r.alias_map.items()if A not in r.seed_names}
		if O:A.append(f"Aliases: {N.join(sorted(O))}")
	if show_defs and r.definitions:
		if not E:A.append(f"\n── Definitions ({len(r.definitions)}) "+'─'*40)
		for C in r.definitions:
			if E:A.append(f"# DEF {C.kind} {C.qname()} {C.file}:{C.lines[0]}-{C.lines[1]}")
			else:A.append(C.display())
			try:X=Path(C.file).read_text(encoding=_M,errors=_N).splitlines();Y=X[C.lines[0]-1:C.lines[1]];A.append('');A.extend(Y);A.append('')
			except OSError:pass
	J=defaultdict(list)
	for B in r.uses:J[B.file].append(B)
	if not E:A.append(f"\n── Uses ({len(r.uses)}) "+'─'*40)
	for G in sorted(J):
		if not E:A.append(f"\n{chr(9473)*60}");A.append(f"  {G}");A.append(chr(9473)*60)
		else:A.append(f"\n# {G}")
		try:H=Path(G).read_text(encoding=_M,errors=_N).splitlines()
		except OSError:
			if not E:
				for B in J[G]:A.append(B.display())
			continue
		Z=J[G];F=[]
		for B in Z:
			P=max(1,B.lines[0]-CONTEXT_LINES);Q=min(len(H),B.lines[1]+CONTEXT_LINES)
			if F and P<=F[-1][1]+1:a,b,c=F[-1];F[-1]=a,max(b,Q),c+[B]
			else:F.append((P,Q,[B]))
		for(I,K,R)in F:
			S=set();L=[]
			for B in R:
				d=f" via {B.via}"if B.via else'';M=f"{B.kind}({B.name}{d}) @ {B.scope}"
				if M not in S:S.add(M);L.append(M)
			T=f"{I}-{K}"if K!=I else str(I)
			if E:
				A.append(f"\n# {T}  |  {V.join(L)}")
				for D in range(I,K+1):A.append(H[D-1]if D<=len(H)else'')
			else:
				A.append(f"\n  # {T}  |  {V.join(L)}");U=set()
				for B in R:
					for D in range(B.lines[0],B.lines[1]+1):U.add(D)
				for D in range(I,K+1):e=H[D-1]if D<=len(H)else'';f='>'if D in U else' ';A.append(f"  {f} {D:4d}  {e}")
	if not E:A.append('')
	return'\n'.join(A)
def main():
	B,C,E,G,H,I,D=_parse_args(sys.argv[1:])
	if not C:
		F=D if D!=DEFAULT_BIND_DEPTH else 1
		if E:print(json.dumps(outline_data(B,depth=F),indent=2))
		else:print(format_outline(B,depth=F))
		return
	A=analyze_multi(B,C,depth=D)
	if not A.definitions and not A.uses:print(f"Nothing found matching {C!r}",file=sys.stderr);sys.exit(1)
	if E:print(json.dumps(A.to_dict(),indent=2))
	else:print(format_results(A,show_defs=H,raw=G))
if __name__=='__main__':main()
