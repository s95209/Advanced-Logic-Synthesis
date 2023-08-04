/*******************************************************************************
+
+  LEDA 5.2  
+
+
+  integer_vector.h
+
+
+  Copyright (c) 1995-2007
+  by Algorithmic Solutions Software GmbH
+  All rights reserved.
+ 
*******************************************************************************/

// $Revision: 1.4 $  $Date: 2007/02/28 07:44:15 $

//---------------------------------------------------------------------
// file generated by notangle from integer_vector.lw
// please debug or modify LEDA web file
// mails and bugs: leda@mpi-sb.mpg.de
// based on LEDA architecture by S. Naeher, C. Uhrig
// coding: K. Mehlhorn, M. Seel
//---------------------------------------------------------------------

#ifndef LEDA_INTEGER_VECTOR_DECL_H
#define LEDA_INTEGER_VECTOR_DECL_H

#if !defined(LEDA_ROOT_INCL_ID)
#define LEDA_ROOT_INCL_ID 520533
#include <LEDA/internal/PREAMBLE.h>
#endif

#include <LEDA/system/basic.h>
#include <LEDA/system/memory.h>

#include <LEDA/numbers/integer.h>
#include <LEDA/numbers/rational.h>

LEDA_BEGIN_NAMESPACE

/*{\Msubst
RTINT integer
}*/

typedef integer RTINT;

#if defined(sgi) && !defined(__GNUC__)
#define RTINT integer
#endif

class __exportC integer_matrix;

/*{\Manpage {integer_vector} {} {Vectors with Integer Entries}}*/

  
  class __exportC integer_vector
  {

  /*{\Mdefinition An instance of data type |integer_vector| is a vector of
  variables of type |integer|, the so called ring type.  Together with the
  type |integer_matrix| it realizes the basic operations of linear
  algebra. Internal correctness tests are executed if compiled with the
  flag \texttt{LA\_SELFTEST}.}*/

    friend class __exportC integer_matrix;

    RTINT* v;
    int d;

    
    inline void allocate_vec_space(RTINT*& vi, int di)
    {
    /* We use this procedure to allocate memory. We use the more efficient
       LEDA scheme. We first get an appropriate piece of memory from the
       memory manager and then initialize each cell by an inplace new. */

      vi = (RTINT*)std_memory.allocate_bytes(di*sizeof(RTINT));
      RTINT* p = vi + di - 1;
      while (p >= vi) { new(p) RTINT;  p--; }   
    }

    inline void deallocate_vec_space(RTINT* vi, int di)
    {
    /* We use this procedure to deallocate memory. We have to free it by
       the LEDA scheme. We first call the destructor for type RTINT for each
       cell of the array and then return the piece of memory to the memory
       manager. */

      RTINT* p = vi + di - 1;
      while (p >= vi)  { p->~RTINT(); p--; }
      std_memory.deallocate_bytes(vi, di*sizeof(RTINT));
      vi = (RTINT*)nil;
    }


    inline void 
    check_dimensions(const integer_vector& vec) const
    { 
      LEDA_PRECOND((d == vec.d), "integer_vector::check_dimensions:\
      object dimensions disagree.")
    }

    // LEDA_MEMORY(integer_vector)

  public:

  /*{\Mcreation v 5}*/

  integer_vector()
  { d = 0; v = (RTINT*)nil; }
  /*{\Mcreate creates an instance |\Mvar| of type |\Mname|. 
              |\Mvar| is initialized to the zero-dimensional vector.}*/

  explicit

  integer_vector(int d); 
  /*{\Mcreate creates an instance |\Mvar| of type |\Mname|. 
              |\Mvar| is initialized to a vector of dimension $d$.}*/ 

  integer_vector(const RTINT& a, const RTINT& b);
  /*{\Mcreate creates an instance |\Mvar| of type |\Mname|.
              |\Mvar| is initialized to the two-dimensional vector $(a,b)$.}*/

  integer_vector(const RTINT& a, const RTINT& b, const RTINT& c);
  /*{\Mcreate creates an instance |\Mvar| of type |\Mname|. 
              |\Mvar| is initialized to the three-dimensional vector 
              $(a,b,c)$.}*/

  integer_vector(const RTINT& a, const RTINT& b, const RTINT& c, const RTINT& d);
  /*{\Mcreate creates an instance |\Mvar| of type |\Mname|; 
              |\Mvar| is initialized to the four-dimensional vector 
              $(a,b,c,d)$.}*/

  integer_vector(const integer_vector&);

  integer_vector& operator=(const integer_vector&);

  ~integer_vector();


  /*{\Moperations 3.3 5.0}*/

  int  dim() const { return d; }
  /*{\Mop       returns the dimension of |\Mvar|.}*/ 

    
  RTINT& operator[](int i)
  /*{\Marrop     returns $i$-th component of |\Mvar|.\\
                 \precond $0\le i \le |v.dim()-1|$. }*/
  { 
    LEDA_OPT_PRECOND((0<=i && i<d), "integer_vector::operator[]: \
    index out of range.")
    return v[i]; 
  }
    
  RTINT operator[](int i) const
  { 
    LEDA_OPT_PRECOND((0<=i && i<d), "integer_vector::operator[]: \
    index out of range.")
    return v[i]; 
  }

  integer_vector& operator+=(const integer_vector& v1);
  /*{\Mbinop     Addition plus assignment.\\
                 \precond |v.dim() == v1.dim()|.}*/

  integer_vector& operator-=(const integer_vector& v1);
  /*{\Mbinop     Subtraction plus assignment.\\
                 \precond |v.dim() == v1.dim()|.}*/
   
  integer_vector  operator+(const integer_vector& v1) const;
  /*{\Mbinop     Addition.\\
                 \precond |v.dim() == v1.dim()|.}*/

  integer_vector  operator-(const integer_vector& v1) const;
  /*{\Mbinop     Subtraction.\\
                 \precond |v.dim() == v1.dim()|.}*/

  RTINT  operator*(const integer_vector& v1) const;
  /*{\Mbinop     Inner Product.\\
                 \precond |v.dim() == v1.dim()|.}*/

  integer_vector  compmul(const RTINT& r) const;

  friend integer_vector operator*(const RTINT& r, const integer_vector& v)
  { return v.compmul(r); } 
  /*{\Mbinopfunc     Componentwise multiplication with number $r$.}*/

  friend integer_vector operator*(const integer_vector& v, const RTINT& r)
  { return v.compmul(r); }
  /*{\Mbinopfunc     Componentwise multiplication with number $r$.}*/

  integer_vector  operator-() const;


  bool     operator==(const integer_vector& w) const;
  bool     operator!=(const integer_vector& w) const 
  { return !(*this == w); }


  friend __exportF  ostream& operator<<(ostream& O, const integer_vector& v);
  /*{\Mbinopfunc   writes |\Mvar| componentwise to the output stream $O$.}*/

  friend __exportF  istream& operator>>(istream& I, integer_vector& v);
  /*{\Mbinopfunc   reads |\Mvar| componentwise from the input stream $I$.}*/


  static int  cmp(const integer_vector&, 
                  const integer_vector&);

  };

  inline int  compare(const integer_vector& x, 
                      const integer_vector& y)
  { return integer_vector::cmp(x,y); }



  /*{\Mimplementation Vectors are implemented by arrays of type
  |integer|. All operations on a vector |v| take time $O(|v.dim()|)$,
  except for |dimension| and $[\ ]$ which take constant time. The space
  requirement is $O(|v.dim()|)$. }*/




#if LEDA_ROOT_INCL_ID == 520533
#undef LEDA_ROOT_INCL_ID
#include <LEDA/internal/POSTAMBLE.h>
#endif


LEDA_END_NAMESPACE

#endif // LEDA_INTEGER_VECTOR_DECL_H


