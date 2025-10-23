
# [strict-aliasing](https://cellperformance.beyond3d.com/articles/2006/06/understanding-strict-aliasing.html)
   - Roughly spealing: Strict aliasing is an assumption, made by the C (or C++) compiler, that dereferencing pointers to objects of **different** types will never refer to the same memory location (i.e. alias eachother.)
   - the exact definition is in [C11 section 6.5 paragraph 7](https://stefansf.de/post/type-based-alias-analysis/)
   - `-fstrict-aliasing/-fno-strict-aliasing` controlls whether this assumption is taken by GCC
   - `-fno-strict-aliasing` will tell GCC to be very conservative since aliasing may happen for pointers of any types.
   - with `-fstrict-aliasing` by default enabled in `-O2`, warning: dereferencing type-punned pointer will break strict-aliasing rules

# [TBAA](https://stefansf.de/post/type-based-alias-analysis/)

 - TBAA(type-based alias analysis) is how compiler enjoy/use **Strict aliasing rules** to optimize.

