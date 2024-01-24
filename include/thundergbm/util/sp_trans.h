/*
* (c) 2017 Virginia Polytechnic Institute & State University (Virginia Tech)   
*                                                                              
*   This program is free software: you can redistribute it and/or modify       
*   it under the terms of the GNU Lesser General Public License Version 2.1.                                  
*                                                                              
*   This program is distributed in the hope that it will be useful,            
*   but WITHOUT ANY WARRANTY; without even the implied warranty of             
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              
*   LICENSE in the root of the repository for details.                 
*                                                                              
*/


#ifndef _SPTRANS_H
#define _SPTRANS_H

#include <omp.h>
#include <algorithm>
#include <numeric>
#include <immintrin.h>

#define NBLOCKS(_n,_bsize) (1 + ((_n)-1)/(_bsize))
#define _SCAN_BSIZE 2048

void scan_kernel_vec_horn(int *in, int *out, int N, bool inclusive, int base) {
#ifdef __AVX2__
    // shrink n to its nearest power of 2
    int log_n = N & 0xfffffff8;
    __m256i param0 = _mm256_set1_epi32(0);
    __m256i param1 = _mm256_set1_epi32(base);
    __m256i param2 = _mm256_set_epi32(6,5,4,3,2,1,0,7);
    __m256i param3 = _mm256_set_epi32(5,4,3,2,1,0,7,6);
    __m256i param4 = _mm256_set_epi32(3,2,1,0,7,6,5,4);
    __m256i param5 = _mm256_set_epi32(7,7,7,7,7,7,7,7);
    if (inclusive) {
        for (int i = 0; i < log_n; i += 8) {
            __m256i va = _mm256_loadu_si256((__m256i *)(&in[i]));
            __m256i vb = _mm256_permutevar8x32_epi32(va, param2);
            __m256i vc = _mm256_insert_epi32(vb, 0, 0);
            __m256i vd = _mm256_add_epi32(va, vc);
            __m256i ve = _mm256_permutevar8x32_epi32(vd, param3);
            __m256i vf = _mm256_blend_epi32(ve, param0, 0x03);
            __m256i vg = _mm256_add_epi32(vd, vf);
            __m256i vh = _mm256_permutevar8x32_epi32(vg, param4);
            __m256i vi = _mm256_blend_epi32(vh, param0, 0x0f);
            __m256i vj = _mm256_add_epi32(vg, vi);
            __m256i vk = _mm256_add_epi32(vj, param1);
            _mm256_storeu_si256((__m256i *)(&out[i]), vk);
            param1 = _mm256_permutevar8x32_epi32(vk, param5);
        }   
        if (log_n < N) {
            out[log_n] = in[log_n] + out[log_n-1];
            for (int i = log_n+1; i < N; i++) {
                out[i] = out[i-1] + in[i];
            }
        }
    } else {
        int sum = in[log_n-1];
        for (int i = 0; i < log_n; i += 8) {
            __m256i va = _mm256_loadu_si256((__m256i *)(&in[i]));
            __m256i vb = _mm256_permutevar8x32_epi32(va, param2);
            __m256i vc = _mm256_insert_epi32(vb, 0, 0);
            __m256i vd = _mm256_add_epi32(va, vc);
            __m256i ve = _mm256_permutevar8x32_epi32(vd, param3);
            __m256i vf = _mm256_blend_epi32(ve, param0, 0x03);
            __m256i vg = _mm256_add_epi32(vd, vf);
            __m256i vh = _mm256_permutevar8x32_epi32(vg, param4);
            __m256i vi = _mm256_blend_epi32(vh, param0, 0x0f);
            __m256i vj = _mm256_add_epi32(vg, vi);
            __m256i vk = _mm256_add_epi32(vj, param1);
            __m256i vl = _mm256_permutevar8x32_epi32(vk, param2);
            __m256i vm = _mm256_blend_epi32(vl, param1, 0x01);
            _mm256_storeu_si256((__m256i *)(&out[i]), vm);
            param1 = _mm256_permutevar8x32_epi32(vk, param5);
        }
        sum += out[log_n-1];
        for (int i = log_n; i < N; i++) {
            int tmp = in[i]; 
            out[i] = sum;
            sum += tmp;
        }
    }
#elif defined __MIC__
    // shrink n to its nearest power of 2
    int log_n = N & 0xfffffff0;
    __m512i param0 = _mm512_set1_epi32(0);
    __m512i param1 = _mm512_set1_epi32(base);
    __m512i param2 = _mm512_set_epi32(14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,15);
    __m512i param3 = _mm512_set_epi32(13,12,11,10,9,8,7,6,5,4,3,2,1,0,15,14);
    __m512i param4 = _mm512_set_epi32(15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15);
    if (inclusive) {
        for (int i = 0; i < log_n; i += 16) {
            __m512i va = _mm512_loadunpacklo_epi32(va, in+i);
                    va = _mm512_loadunpackhi_epi32(va, in+i+16);
            __m512i vb = _mm512_mask_permutevar_epi32(param0, 0xfffe, param2, va);
            __m512i vc = _mm512_add_epi32(va,vb);
            __m512i vd = _mm512_mask_permutevar_epi32(param0, 0xfffc, param3, vc);
            __m512i ve = _mm512_add_epi32(vc,vd);
            __m512i vf = _mm512_mask_permute4f128_epi32(param0, 0xfff0, ve, _MM_PERM_CBAD);
            __m512i vg = _mm512_add_epi32(ve,vf);
            __m512i vh = _mm512_mask_permute4f128_epi32(param0, 0xff00, vg, _MM_PERM_BADC);
            __m512i vi = _mm512_add_epi32(vg,vh);
            __m512i vj = _mm512_add_epi32(vi, param1);
            _mm512_packstorelo_epi32(out+i, vj);
            _mm512_packstorehi_epi32(out+i+16, vj);
            param1 = _mm512_permutevar_epi32(param4, vj);
        }   
        if (log_n < N) {
            out[log_n] = in[log_n] + out[log_n-1];
            for (int i = log_n+1; i < N; i++) {
                out[i] = out[i-1] + in[i];
            }
        }
    } else {
        int sum = in[log_n-1];
        for (int i = 0; i < log_n; i += 16) {
            __m512i va = _mm512_loadunpacklo_epi32(va, in+i);
                    va = _mm512_loadunpackhi_epi32(va, in+i+16);
            __m512i vb = _mm512_mask_permutevar_epi32(param0, 0xfffe, param2, va);
            __m512i vc = _mm512_add_epi32(va,vb);
            __m512i vd = _mm512_mask_permutevar_epi32(param0, 0xfffc, param3, vc);
            __m512i ve = _mm512_add_epi32(vc,vd);
            __m512i vf = _mm512_mask_permute4f128_epi32(param0, 0xfff0, ve, _MM_PERM_CBAD);
            __m512i vg = _mm512_add_epi32(ve,vf);
            __m512i vh = _mm512_mask_permute4f128_epi32(param0, 0xff00, vg, _MM_PERM_BADC);
            __m512i vi = _mm512_add_epi32(vg,vh);
            __m512i vj = _mm512_add_epi32(vi, param1);
            __m512i vk = _mm512_mask_permutevar_epi32(param1, 0xfffe, param2, vj);
            _mm512_packstorelo_epi32(out+i, vk);
            _mm512_packstorehi_epi32(out+i+16, vk);
            param1 = _mm512_permutevar_epi32(param4, vj);
        }
        sum += out[log_n-1];
        for (int i = log_n; i < N; i++) {
            int tmp = in[i]; 
            out[i] = sum;
            sum += tmp;
        }
    }
#endif
}

void scan_kernel(int *in, int *out, int N, bool inclusive, int base) {
#if defined(__AVX2__) || defined(__MIC__)
    scan_kernel_vec_horn(in, out, N, inclusive, base);
#else
    if (inclusive) {
        out[0] = in[0] + base;
        for(int i = 1; i < N; i++) {
            out[i] = out[i-1] + in[i];
        }
    } else {
        int sum = base;
        for(int i = 0; i < N; i++) {
            int tmp = in[i];
            out[i] = sum;
            sum += tmp;
        }
    }
    return;
#endif
}

void scan_omp(int *in, int *out, int N, bool inclusive, int base) {
    int p = 1;
#pragma omp parallel
    p = omp_get_num_threads();
    int nblocks = NBLOCKS(N, _SCAN_BSIZE);
    if (nblocks <= 2) {
        scan_kernel(in, out, N, inclusive, base);
        return;
    }
    int *sums = new int[nblocks];
#pragma omp parallel for 
    for (int i = 0; i < nblocks; i++) {
        int st = i * _SCAN_BSIZE;
        int ed = std::min(st + _SCAN_BSIZE, N);
        int sum = std::accumulate(in+st, in+ed, 0);
        sums[i] = sum;
    }
    scan_omp(sums, sums, nblocks, false, base);
#pragma omp parallel for 
    for(int i = 0; i < nblocks; i++) {
        int st = i * _SCAN_BSIZE;
        int ed = std::min(st + _SCAN_BSIZE, N);
        scan_kernel(in+st, out+st, ed-st, inclusive, sums[i]);
    }
    delete []sums;
}

template<typename iT, typename vT>
void sptrans_scanTrans(int  m,
                       int  n,
                       int  nnz,
                       iT  *csrRowPtr,
                       iT  *csrColIdx,
                       vT  *csrVal,
                       iT  *cscRowIdx,
                       iT  *cscColPtr,
                       vT  *cscVal)
{
    int i, j, k, ii, jj, procs; 

#pragma omp parallel
    procs = omp_get_num_threads();
    
    int size = nnz / procs;
    int left = nnz % procs;
    int *csrRowUnroll = (int *)malloc(nnz * sizeof(int));
    int *intra        = (int *)malloc(nnz * sizeof(int));
    int *wb_index     = (int *)malloc(nnz * sizeof(int));
    int *inter        = (int *)malloc((long long)(procs + 1) * n * sizeof(int));
    memset(inter, 0, (long long)(procs + 1) * n * sizeof(int));
#pragma omp parallel default(shared) private(i)
    {
        size_t index = 0;
        int tid = omp_get_thread_num();
        size_t inter_start = n + (long long)tid * n;
        size_t intra_start = size * tid;
        int len = size;
        if (left != 0) {
            if (tid < left) {
                len += 1;
                intra_start = tid * size + tid;
            } else {
                intra_start = tid * size + left;
            }
        }
        for (i = 0; i < len; i++) {
            index = inter_start + csrColIdx[intra_start + i];
            intra[intra_start + i] = inter[index];
            inter[index]++;
        }
    }
#pragma omp parallel for default(shared) private(i, j) //schedule(dynamic)
    for (i = 0; i < n; i++) {
        for (j = 2; j < procs + 1; j++) {
            inter[i + n * j] += inter[i + n * (j - 1)];
        }
    }
#pragma omp parallel for default(shared) private(i) //schedule(dynamic)
#pragma ivdep
    for (i = 0; i < n; i++) {
        cscColPtr[i + 1] = inter[(long long)n * procs + i];
    }
    scan_omp(cscColPtr,cscColPtr,n+1,true,0);
#pragma omp parallel for default(shared) private(i, j, ii, jj) //schedule(dynamic)
    for (i = 0 ; i < m; i++) {
        ii = csrRowPtr[i + 1] - csrRowPtr[i];
        jj = csrRowPtr[i];
        for (j = 0; j < ii; j++) {
            csrRowUnroll[jj + j] = i;
        }
    }

#pragma omp parallel default(shared) private(i, j)
    {
        size_t tid = omp_get_thread_num();
        size_t inter_start = (long long)tid * n; 
        size_t intra_start = size * (long long)tid;
        int len = size;
        size_t colIdx, offset;

        if (left != 0) {
            if (tid < left) {
                len += 1;
                intra_start = tid * size + tid;
            } else {
                intra_start = tid * size + left;
            }
        }

        for (i = 0; i < len; i++) {
            colIdx = csrColIdx[intra_start + i];
            offset = cscColPtr[colIdx] + inter[inter_start + colIdx] + intra[intra_start + i];
            wb_index[intra_start + i] = offset;
        }
    }

    if (csrVal != NULL) {
#pragma omp parallel default(shared) private(i, j)
        {
            size_t offset = 0;
            size_t tid = (long long)omp_get_thread_num();
            size_t inter_start = tid * n; 
            size_t intra_start = size * tid;
            int len = size;

            if (left != 0) {
                if (tid < left) {
                    len += 1;
                    intra_start = tid * size + tid;
                } else {
                    intra_start = tid * size + left;
                }
            }

            for (i = 0; i < len; i++) {
                offset            = wb_index[intra_start + i];
                cscVal[offset]    = csrVal[intra_start + i];
                cscRowIdx[offset] = csrRowUnroll[intra_start + i];
            }
        }
    } else {
#pragma omp parallel default(shared) private(i, j)
        {
            size_t offset = 0;
            size_t tid = (long long)omp_get_thread_num();
            size_t inter_start = tid * n; 
            size_t intra_start = size * tid;
            int len = size;

            if (left != 0) {
                if (tid < left) {
                    len += 1;
                    intra_start = tid * size + tid;
                } else {
                    intra_start = tid * size + left;
                }
            }

            for(i = 0; i < len; i++) {
                offset            = wb_index[intra_start + i];
                cscRowIdx[offset] = csrRowUnroll[intra_start + i];
            }
        }
    }

    free(csrRowUnroll);
    free(inter);
    free(intra);
    free(wb_index);
}

// merge two csr by one thread
template<typename iT, typename vT>
void segment_merge_csr(int len,
                       iT *csrRowPtrA, 
                       iT *csrColIdxA, 
                       vT *csrValA, 
                       iT *csrRowPtrB, 
                       iT *csrColIdxB, 
                       vT *csrValB,
                       iT *csrRowPtrC,
                       iT *csrColIdxC,
                       vT *csrValC)
{
    int i, j, k;

    for (i = 0; i < len; i++) {
        csrRowPtrC[i] = csrRowPtrA[i] + csrRowPtrB[i];
    }

    for (i = 0; i < len - 1; i++) {
        int starta = csrRowPtrA[i];
        int startb = csrRowPtrB[i];
        int startc = csrRowPtrC[i];
        int lena   = csrRowPtrA[i + 1] - csrRowPtrA[i];
        int lenb   = csrRowPtrB[i + 1] - csrRowPtrB[i];
        int lenc   = csrRowPtrC[i + 1] - csrRowPtrC[i];
        
        for (j = 0; j < lena; j++) {
            csrColIdxC[startc + j] = csrColIdxA[starta + j];
            csrValC[startc + j]    = csrValA[starta + j];
        }
        for (k = 0; k < lenb; k++) {
            csrColIdxC[startc + lena + k] = csrColIdxB[startb + k];
            csrValC[startc + lena + k]    = csrValB[startb + k];
        }
    }
}

// merge multiple csr by one thread
template<typename iT, typename vT>
void segment_merge_multiple_csr(int ncsr,
                                int len,
                                iT *csrRowPtrA, 
                                iT *csrColIdxA, 
                                vT *csrValA, 
                                iT *csrRowPtrB, 
                                iT *csrColIdxB, 
                                vT *csrValB)
{
    int i, j, k;

    for (i = 0; i < len; i++) {
        for (j = 0; j < ncsr; j++) {
            csrRowPtrB[i] += csrRowPtrA[i + len * j];
        }
    }

    for (i = 0; i < len - 1; i++) {
        int startb  = csrRowPtrB[i];
        int starta  = 0;
        int lena    = 0;
        int offseta = 0;
        int offsetb = 0;
        
        for (j = 0; j < ncsr; j++) {
            starta   = csrRowPtrA[i + j * len]; 
            lena     = csrRowPtrA[i + 1 + j * len] - csrRowPtrA[i + j * len];
            offseta += csrRowPtrA[j * (len - 1)]; 

            for (k = 0; k < lena; k++) {
                csrColIdxB[startb + offsetb + k] = csrColIdxA[offseta + starta + k];
                csrValB[startb + offsetb + k]    = csrValA[offseta + starta + k];
            }
            offsetb += lena;
        }
    }
}

// merge two csr by multiple threads
template<typename iT, typename vT>
void segment_merge_csr_multithreads(int begin,
                                    int end,
                                    iT *csrRowPtrA, 
                                    iT *csrColIdxA, 
                                    vT *csrValA, 
                                    iT *csrRowPtrB, 
                                    iT *csrColIdxB, 
                                    vT *csrValB,
                                    iT *csrRowPtrC,
                                    iT *csrColIdxC,
                                    vT *csrValC)
{
    int i, j, k, tid, procs;
    tid   = omp_get_thread_num();
    procs = omp_get_num_threads();

    for (i = begin; i <= end; i++) {
        csrRowPtrC[i] = csrRowPtrA[i] + csrRowPtrB[i];
    }

    for (i = begin; i < end; i++) {
        int starta = csrRowPtrA[i];
        int startb = csrRowPtrB[i];
        int startc = csrRowPtrC[i];
        int lena   = csrRowPtrA[i + 1] - csrRowPtrA[i];
        int lenb   = csrRowPtrB[i + 1] - csrRowPtrB[i];
        int lenc   = csrRowPtrC[i + 1] - csrRowPtrC[i];
        
        for (j = 0; j < lena; j++) {
            csrColIdxC[startc + j] = csrColIdxA[starta + j];
            csrValC[startc + j]    = csrValA[starta + j];
        }
        for (k = 0; k < lenb; k++) {
            csrColIdxC[startc + lena + k] = csrColIdxB[startb + k];
            csrValC[startc + lena + k]    = csrValB[startb + k];
        }
    }

    if (tid != procs - 1) {
        int starta = csrRowPtrA[end];
        int startb = csrRowPtrB[end];
        int startc = csrRowPtrA[end] + csrRowPtrB[end];
        int lena   = csrRowPtrA[end + 1] - csrRowPtrA[end];
        int lenb   = csrRowPtrB[end + 1] - csrRowPtrB[end];
        int lenc   = lena + lenb;
        
        for (j = 0; j < lena; j++) {
            csrColIdxC[startc + j] = csrColIdxA[starta + j];
            csrValC[startc + j]    = csrValA[starta + j];
        }
        for (k = 0; k < lenb; k++) {
            csrColIdxC[startc + lena + k] = csrColIdxB[startb + k];
            csrValC[startc + lena + k]    = csrValB[startb + k];
        }
    }
}

template<typename iT, typename vT>
void sptrans_mergeTrans(int  m,
                        int  n,
                        int  nnz,
                        iT  *csrRowPtr,
                        iT  *csrColIdx,
                        vT  *csrVal,
                        iT  *cscRowIdx,
                        iT  *cscColPtr,
                        vT  *cscVal)
{
    int i, j, k, ii, jj, kk, procs;
 
#pragma omp parallel
    procs = omp_get_num_threads();

    // bsize can be configured to LLC size per core
    int bsize = nnz / procs;
    int blocksByL2 = procs;
    int dataLeftL2 = nnz % procs;
    int nthreadPerBlock = 1;

#if defined(__MIC__) || defined(__AVX2__)
    char *env_nthreadPerBlock;
    if ((env_nthreadPerBlock = getenv("NTHREADPERBLOCK")) != NULL) {
        nthreadPerBlock = atoi(env_nthreadPerBlock);
    } else {
        nthreadPerBlock = 1;
    }
    bsize = nnz / (procs / nthreadPerBlock);
    blocksByL2 = procs / nthreadPerBlock;
    dataLeftL2 = nnz % (procs / nthreadPerBlock);
#endif

    long allocBlocks = blocksByL2 != 0 ? blocksByL2 : (dataLeftL2 != 0 ? 1 : 0);
    bsize = blocksByL2 != 0 ? bsize : (dataLeftL2 != 0 ? dataLeftL2 : 0);
    long size = 2 * allocBlocks * (n + 1) * sizeof(iT);
    iT *csrRowUnroll = (iT *)malloc(nnz * sizeof(iT));
    iT *cscColPtrAux = (iT *)malloc(size);
    iT *cscRowIdxAux = (iT *)malloc(2 * nnz * sizeof(iT));
    vT *cscValAux    = (vT *)malloc(2 * nnz * sizeof(vT));

#ifndef __MIC__
    memset(cscColPtrAux, 0, size);
#endif

    iT *ptra, *ptrb, *idxa, *idxb;
    vT *vala, *valb;
    ptra = cscColPtrAux;
    ptrb = cscColPtrAux + allocBlocks * (n + 1);
    idxa = cscRowIdxAux;
    idxb = cscRowIdxAux + nnz;
    vala = cscValAux;
    valb = cscValAux + nnz;

#pragma omp parallel for default(shared) private(i, j, ii, jj)
    for (i = 0 ; i < m; i++) {
        ii = csrRowPtr[i + 1] - csrRowPtr[i];
        jj = csrRowPtr[i];
        for (j = 0; j < ii; j++) 
        {
            csrRowUnroll[jj + j] = i;
        }
    }

    // Stage 1: csc to csr for each tile
    int loop = blocksByL2 / procs;
    int left = blocksByL2 % procs;

    if (csrVal != NULL) {
#pragma omp parallel default(shared) private(i, j, k)
        {
            int tid = omp_get_thread_num();
            for (i = 0; i < loop; i++) {
                long inter_start = i * procs * (n + 1) + tid * (n + 1);
                long intra_start = i * procs * bsize + tid * bsize;
#ifdef __MIC__
                for (j = 0; j < n + 1; j++) {
                    ptrb[inter_start + j] = 0;
                    ptra[inter_start + j] = 0;
                }
#endif
                for (j = 0; j < bsize; j++) {
                    int col = csrColIdx[intra_start + j];
                    ptrb[inter_start + col + 1]++;
                    ptra[inter_start + col + 1]++;
                }
                for (k = 0; k < n; k++) {
                    int tmp = ptrb[inter_start + k + 1];
                    ptra[inter_start + k + 1] = tmp + ptra[inter_start + k];
                }
                for (j = 0; j < bsize; j++) {
                    int  col = csrColIdx[intra_start + j];
                    long pos = intra_start + ptra[inter_start + col + 1] - ptrb[inter_start + col + 1]--;
                    idxa[pos] = csrRowUnroll[intra_start + j];
                    vala[pos] = csrVal[intra_start + j];
                }
            }

            if (left) {
#if defined(__MIC__) || defined(__AVX2__)
                if (tid % nthreadPerBlock == 0)
#else
                if (tid < left) 
#endif
                {
                    long inter_start = loop * procs * (n + 1) + (tid / nthreadPerBlock) * (n + 1);
                    long intra_start = loop * procs * bsize + (tid / nthreadPerBlock) * bsize;
#ifdef __MIC__
                    for (j = 0; j < n + 1; j++) {
                        ptrb[inter_start + j] = 0;
                        ptra[inter_start + j] = 0;
                    }
#endif
                    for (j = 0; j < bsize; j++) {
                        int col = csrColIdx[intra_start + j];
                        ptrb[inter_start + col + 1]++;
                    }
                    for (k = 0; k < n; k++) {
                        int tmp = ptrb[inter_start + k + 1];
                        ptra[inter_start + k + 1] = tmp + ptra[inter_start + k];
                    }
                    for (j = 0; j < bsize; j++) {
                        int  col = csrColIdx[intra_start + j];
                        long pos = intra_start + ptra[inter_start + col + 1] - ptrb[inter_start + col + 1]--;
                        idxa[pos] = csrRowUnroll[intra_start + j];
                        vala[pos] = csrVal[intra_start + j];
                    }
                }
            }
        }
    } else {
#pragma omp parallel default(shared) private(i, j, k)
        {
            int tid = omp_get_thread_num();
            for (i = 0; i < loop; i++) {
                long inter_start = i * procs * (n + 1) + tid * (n + 1);
                long intra_start = i * procs * bsize + tid * bsize;
#ifdef __MIC__
                for (j = 0; j < n + 1; j++) {
                    ptrb[inter_start + j] = 0;
                    ptra[inter_start + j] = 0;
                }
#endif
                for (j = 0; j < bsize; j++) {
                    int col = csrColIdx[intra_start + j];
                    ptrb[inter_start + col + 1]++;
                    ptra[inter_start + col + 1]++;
                }
                for (k = 0; k < n; k++) {
                    int tmp = ptrb[inter_start + k + 1];
                    ptra[inter_start + k + 1] = tmp + ptra[inter_start + k];
                }
                for (j = 0; j < bsize; j++) {
                    int  col = csrColIdx[intra_start + j];
                    long pos = intra_start + ptra[inter_start + col + 1] - ptrb[inter_start + col + 1]--;
                    idxa[pos] = csrRowUnroll[intra_start + j];
                }
            }

            if (left) {
#if defined(__MIC__) || defined(__AVX2__)
                if (tid % nthreadPerBlock == 0)
#else
                if (tid < left) 
#endif
                {
                    long inter_start = loop * procs * (n + 1) + (tid / nthreadPerBlock) * (n + 1);
                    long intra_start = loop * procs * bsize + (tid / nthreadPerBlock) * bsize;
#ifdef __MIC__
                    for (j = 0; j < n + 1; j++) {
                        ptrb[inter_start + j] = 0;
                        ptra[inter_start + j] = 0;
                    }
#endif
                    for (j = 0; j < bsize; j++) {
                        int col = csrColIdx[intra_start + j];
                        ptrb[inter_start + col + 1]++;
                    }
                    for (k = 0; k < n; k++) {
                        int tmp = ptrb[inter_start + k + 1];
                        ptra[inter_start + k + 1] = tmp + ptra[inter_start + k];
                    }
                    for (j = 0; j < bsize; j++) {
                        int  col = csrColIdx[intra_start + j];
                        long pos = intra_start + ptra[inter_start + col + 1] - ptrb[inter_start + col + 1]--;
                        idxa[pos] = csrRowUnroll[intra_start + j];
                    }
                }
            }
        }
    }
   
    if (dataLeftL2) {
        long inter_start = 0, intra_start = 0, len = dataLeftL2;
        if (blocksByL2) {
            inter_start = (blocksByL2 - 1) * (n + 1);
            intra_start = (blocksByL2 - 1) * bsize;
            len += bsize;
        }
        for (k = 0; k < n; k++) {
            ptra[inter_start + k + 1] = 0;
        }
        for (j = 0; j < len; j++) {
            int col = csrColIdx[intra_start + j];
            ptrb[inter_start + col + 1]++;
        }
        for (k = 0; k < n; k++) {
            long tmp = ptrb[inter_start + k + 1];
            ptra[inter_start + k + 1] = tmp + ptra[inter_start + k];
        }
        if (csrVal != NULL) {
            for (j = 0; j < len; j++) {
                int col = csrColIdx[intra_start + j];
                long pos = intra_start + ptra[inter_start + col + 1] - ptrb[inter_start + col + 1]--;
                idxa[pos] = csrRowUnroll[intra_start + j];
                vala[pos] = csrVal[intra_start + j];
            }
        } else {
            for (j = 0; j < len; j++) {
                int col = csrColIdx[intra_start + j];
                long pos = intra_start + ptra[inter_start + col + 1] - ptrb[inter_start + col + 1]--;
                idxa[pos] = csrRowUnroll[intra_start + j];
            }
        }
    }

    // Stage 2: in-memory merge csc until number of blocks is less than twice of thread number 
    int blocks = allocBlocks;
    while (blocks >= 2 * procs) {
        loop = blocks / (2 * procs);
        left = blocks % (2 * procs);

#pragma omp parallel default(shared) private(i, j, k)
        {
            int tid = omp_get_thread_num();

            iT *csrRowPtrA, *csrRowPtrB, *csrRowPtrC;
            iT *csrColIdxA, *csrColIdxB, *csrColIdxC;
            vT *csrValA, *csrValB, *csrValC;
         
            iT *lptra = ptra;
            iT *lptrb = ptrb;
            iT *lidxa = idxa;
            iT *lidxb = idxb;
            vT *lvala = vala;
            vT *lvalb = valb;
   
            for (i = 0; i < loop; i++) {
                int srcBlockStart = i * (2 * procs) + 2 * tid;
                int dstBlockStart = i * procs + tid;
                
                csrRowPtrA = lptra + srcBlockStart * (n + 1);
                csrColIdxA = lidxa + srcBlockStart * bsize;
                csrValA    = lvala + srcBlockStart * bsize;
                
                csrRowPtrB = lptra + (srcBlockStart + 1) * (n + 1);
                csrColIdxB = lidxa + (srcBlockStart + 1) * bsize;
                csrValB    = lvala + (srcBlockStart + 1) * bsize;
                
                csrRowPtrC = lptrb + dstBlockStart * (n + 1);
                csrColIdxC = lidxb + srcBlockStart * bsize;
                csrValC    = lvalb + srcBlockStart * bsize;
                
                segment_merge_csr(n + 1, csrRowPtrA, csrColIdxA, csrValA,
                                         csrRowPtrB, csrColIdxB, csrValB,
                                         csrRowPtrC, csrColIdxC, csrValC);
            }
#pragma omp single
            {
                blocks = loop * procs;
            }

            if (left) {
                if (tid < (left / 2)) {
                    int srcBlockStart = loop * (2 * procs) + 2 * tid;
                    int dstBlockStart = loop * procs + tid;
                 
                    csrRowPtrA = lptra + srcBlockStart * (n + 1);
                    csrColIdxA = lidxa + srcBlockStart * bsize;
                    csrValA    = lvala + srcBlockStart * bsize;
                
                    csrRowPtrB = lptra + (srcBlockStart + 1) * (n + 1);
                    csrColIdxB = lidxa + (srcBlockStart + 1) * bsize;
                    csrValB    = lvala + (srcBlockStart + 1) * bsize;

                    csrRowPtrC = lptrb + dstBlockStart * (n + 1);
                    csrColIdxC = lidxb + srcBlockStart * bsize;
                    csrValC    = lvalb + srcBlockStart * bsize;
                    
                    segment_merge_csr(n + 1, csrRowPtrA, csrColIdxA, csrValA,
                                             csrRowPtrB, csrColIdxB, csrValB,
                                             csrRowPtrC, csrColIdxC, csrValC);
                }
#pragma omp single
                {
                    blocks += left / 2;
                    if (left % 2) {
                        int srcBlockStart = loop * (2 * procs) + left - 1;
                        int dstBlockStart = blocks;
                        memcpy(lptrb + dstBlockStart * (n + 1), lptra + srcBlockStart * (n + 1), (n + 1) * sizeof(iT));
                        memcpy(lidxb + srcBlockStart * bsize,   lidxa + srcBlockStart * bsize,   lptra[srcBlockStart * (n + 1) + n] * sizeof(iT));
                        memcpy(lvalb + srcBlockStart * bsize,   lvala + srcBlockStart * bsize,   lptra[srcBlockStart * (n + 1) + n] * sizeof(vT));
                        blocks += 1;
                    }
                }
            }
        }

        iT *ptrt, *idxt;
        vT *valt;

        ptrt = ptra;
        ptra = ptrb;
        ptrb = ptrt;
        idxt = idxa;
        idxa = idxb;
        idxb = idxt;
        valt = vala;
        vala = valb;
        valb = valt;
    
        bsize *= 2;
    }

    // Stage 3: use multiple threads to merge each pair until one csc
    while (blocks != 1) {
        int pairs = blocks / 2;
        int lefts = blocks % 2;
        int nthreadPerPair = procs / pairs;

#pragma omp parallel default(shared) private(i, j, k)
        {
            int tid = omp_get_thread_num();

            iT *csrRowPtrA, *csrRowPtrB, *csrRowPtrC;
            iT *csrColIdxA, *csrColIdxB, *csrColIdxC;
            vT *csrValA, *csrValB, *csrValC;

            iT *lptra = ptra;
            iT *lptrb = ptrb;
            iT *lidxa = idxa;
            iT *lidxb = idxb;
            vT *lvala = vala;
            vT *lvalb = valb;

            int srcBlockStart, dstBlockStart;
            int rowPerThread, rowLeftPerId, mapTRPerPair, rowBeginThread, rowEndThread; 
             
            if (tid < pairs * nthreadPerPair) {
                if (nthreadPerPair == 1) {
                    srcBlockStart = tid * 2;
                    dstBlockStart = tid;
             
                    csrRowPtrA = lptra + srcBlockStart * (n + 1);
                    csrColIdxA = lidxa + srcBlockStart * bsize;
                    csrValA    = lvala + srcBlockStart * bsize;
                    
                    csrRowPtrB = lptra + (srcBlockStart + 1) * (n + 1);
                    csrColIdxB = lidxa + (srcBlockStart + 1) * bsize;
                    csrValB    = lvala + (srcBlockStart + 1) * bsize;
                 
                    csrRowPtrC = lptrb + dstBlockStart * (n + 1);
                    csrColIdxC = lidxb + srcBlockStart * bsize;
                    csrValC    = lvalb + srcBlockStart * bsize;
             
                    segment_merge_csr(n + 1, csrRowPtrA, csrColIdxA, csrValA,
                                             csrRowPtrB, csrColIdxB, csrValB,
                                             csrRowPtrC, csrColIdxC, csrValC);
                } else {
                    srcBlockStart = (tid / nthreadPerPair) * 2;
                    dstBlockStart = tid / nthreadPerPair;

                    rowPerThread = (n + 1) / nthreadPerPair;
                    rowLeftPerId = (n + 1) % nthreadPerPair;
                    mapTRPerPair = tid % nthreadPerPair;
                    rowBeginThread = mapTRPerPair * rowPerThread;
                    rowEndThread   = rowBeginThread + rowPerThread - 1; 
             
                    if (rowLeftPerId) {
                        if (mapTRPerPair < rowLeftPerId) {
                            rowBeginThread = rowBeginThread + mapTRPerPair;
                            rowEndThread += (mapTRPerPair + 1);
                        } else {
                            rowBeginThread = rowBeginThread + rowLeftPerId;
                            rowEndThread += rowLeftPerId;
                        }
                    }

                    csrRowPtrA = lptra + srcBlockStart * (n + 1);
                    csrColIdxA = lidxa + srcBlockStart * bsize;
                    csrValA    = lvala + srcBlockStart * bsize;
                    
                    csrRowPtrB = lptra + (srcBlockStart + 1) * (n + 1);
                    csrColIdxB = lidxa + (srcBlockStart + 1) * bsize;
                    csrValB    = lvala + (srcBlockStart + 1) * bsize;
                 
                    csrRowPtrC = lptrb + dstBlockStart * (n + 1);
                    csrColIdxC = lidxb + srcBlockStart * bsize;
                    csrValC    = lvalb + srcBlockStart * bsize;
             
                    segment_merge_csr_multithreads(rowBeginThread, rowEndThread, csrRowPtrA, csrColIdxA, csrValA,
                                                   csrRowPtrB, csrColIdxB, csrValB,
                                                   csrRowPtrC, csrColIdxC, csrValC);
                }
            }

            if (lefts) {
#ifdef __MIC__
#pragma omp for private(i)
                for (i = 0; i < n + 1; i++) {
                    lptrb[pairs * (n + 1) + i] = lptra[(blocks - 1) * (n + 1) + i]; 
                }
#pragma omp for private(i)
                for (i = 0; i < lptra[(blocks - 1) * (n + 1) + n]; i++) {
                    lidxb[(blocks - 1) * bsize + i] = lidxa[(blocks - 1) * bsize + i];
                    lvalb[(blocks - 1) * bsize + i] = lvala[(blocks - 1) * bsize + i];
                }
#pragma omp single
                {
                    blocks = pairs + 1;
                }
#else
#pragma omp single
                {
                    int srcBlockStart = blocks - 1;
                    int dstBlockStart = pairs;
                    memcpy(lptrb + dstBlockStart * (n + 1), lptra + srcBlockStart * (n + 1), (n + 1) * sizeof(iT));
                    memcpy(lidxb + srcBlockStart * bsize,   lidxa + srcBlockStart * bsize,   lptra[srcBlockStart * (n + 1) + n] * sizeof(iT));
                    memcpy(lvalb + srcBlockStart * bsize,   lvala + srcBlockStart * bsize,   lptra[srcBlockStart * (n + 1) + n] * sizeof(vT));
                    blocks = pairs + 1;
                }
#endif
            } else {
#pragma omp single
                {
                    blocks = pairs;
                }
            }
        }
        
        iT *ptrt, *idxt;
        vT *valt;

        ptrt = ptra;
        ptra = ptrb;
        ptrb = ptrt;
        idxt = idxa;
        idxa = idxb;
        idxb = idxt;
        valt = vala;
        vala = valb;
        valb = valt;
    
        bsize *= 2;
    }

#ifdef __MIC__
#pragma omp parallel for default(shared) private(i)
    for (i = 0; i < n + 1; i++) {
        cscColPtr[i] = ptra[i];
    }
    if (csrVal != NULL) {
#pragma omp parallel for default(shared) private(i)
        for (i = 0; i < ptra[n]; i++) {
            cscRowIdx[i] = idxa[i];
            cscVal[i]    = vala[i];
        }
    } else {
#pragma omp parallel for default(shared) private(i)
        for (i = 0; i < ptra[n]; i++) {
            cscRowIdx[i] = idxa[i];
        }
    }
#else
    memcpy(cscColPtr, ptra, (n + 1) * sizeof(iT));
    memcpy(cscRowIdx, idxa, (ptra[n] - ptra[0]) * sizeof(iT));
    if(csrVal != NULL)
        memcpy(cscVal,    vala, (ptra[n] - ptra[0]) * sizeof(vT));
#endif

    free(csrRowUnroll);
    free(cscColPtrAux);
    free(cscRowIdxAux);
    free(cscValAux);
}
#endif
