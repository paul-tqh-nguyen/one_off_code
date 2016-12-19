
import Monad( MonadPlus( .. ), guard )

newtype StateTransEx s a = STE( s -> Maybe (s, a) )

data Maybe t = Just t | Nothing

instance Monad (StateTransEx s)
  where
    -- (>>=) :: StateTransEx s a -> (a -> StateTransEx s b) -> StateTransEx s b
    (STE p) >>= k = STE( \s0 -> case p s0 of
                                 Just (s1, a) -> let (STE q) = k a
                                                 in q s1
                                 Nothing -> Nothing)
                
    -- return :: a -> StateTransEx s a
    return a = STE( \s -> Just (s, a) )



instance MonadPlus (StateTransEx s)
  where
    -- mzero :: StateTransEx s a
    mzero = STE( \s -> Nothing )
    -- mplus ::  StateTransEx s a ->  StateTransEx s a ->  StateTransEx s a
    (STE p) `mplus` (STE q)  =  STE( \s0 -> case p s0 of
    					Just (s1, a) -> Just (s1, a)
    					Nothing -> q s0 )

applySTE (STE p) s = p s

type QState = ([Int], [Int], [Int])

getCols = STE( \(cols, swDiags, seDiags) ->
                Just ((cols, swDiags, seDiags), cols) )
getSWDiags = STE( \(cols, swDiags, seDiags) ->
                  Just ((cols, swDiags, seDiags), swDiags) )
getSEDiags = STE( \(cols, swDiags, seDiags) ->
                  Just ((cols, swDiags, seDiags), seDiags) )

putCols c = STE( \(cols, swDiags, seDiags) ->
                  Just ((c:cols, swDiags, seDiags), ()) )
putSWDiags sw = STE( \(cols, swDiags, seDiags) ->
                      Just ((cols, sw:swDiags, seDiags), ()) )
putSEDiags se = STE( \(cols, swDiags, seDiags) ->
                      Just ((cols, swDiags, se:seDiags), ()) )

guard true = return()
guard false = mzero

tryPutCol c =
	do cols <- getCols
	   guard (c `notElem` cols)
	   putCols c
	   
tryPutSWDiag sw =
	do swDiags <- getSWDiags
	   guard (sw `notElem` swDiags)
	   putSWDiags sw
	   
tryPutSEDiag se =
	do seDiags <- getSEDiags
	   guard (se `notElem` seDiags)
	   putSEDiags se

place r c = 
	do tryPutCol c
	   tryPutSWDiag (c-r)
	   tryPutSEDiag (c+r)

queens r colNum =
	if r == 0
	then getCols             -- Success, return list of columns
	else tryEach [0..colNum-1] (\c ->
		do place (r-1) c
		   queens (r-1) colNum )

tryEach :: MonadPlus m => [a] -> (a -> m b) -> m b   
tryEach [] f = mzero
tryEach (h:t) f = f h `mplus` tryEach t f

applySTE (queens 8 8) ([], [], [])






















