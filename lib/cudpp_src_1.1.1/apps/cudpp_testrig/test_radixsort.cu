// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include <stdio.h>
#include <cutil.h>
#include <math.h>

#include "cudpp.h"
#include "cudpp_testrig_options.h"

template <typename T>
class SortSupport
{
public:
    static void fillVector(T *a, size_t numElements, unsigned int keybits) {}
    static int  verifySort(T *keysSorted, unsigned int *valuesSorted, T *keysUnsorted, size_t len) { return 0; }
};

template<>
void SortSupport<unsigned int>::fillVector(unsigned int *a, size_t numElements, unsigned int keybits)
{
    // Fill up with some random data
    int keyshiftmask = 0;
    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
    int keymask = 0xffff;
    if (keybits < 16) keymask = (1 << keybits) - 1;            

    srand(95123);
    for(unsigned int i=0; i < numElements; ++i)   
    { 
        a[i] = ((rand() & keyshiftmask)<<16) | (rand() & keymask); 
    }
}

template<>
void SortSupport<float>::fillVector(float *a, size_t numElements, unsigned int keybits)
{
    srand(95123);
    for(unsigned int j = 0; j < numElements; j++)
    {
        a[j] = pow(-1,(float)j)*(float)((rand()<<16) | rand());          
    }
}

// assumes the values were initially indices into the array, for simplicity of 
// checking correct order of values
template<>
int SortSupport<unsigned int>::verifySort(unsigned int *keysSorted, unsigned int *valuesSorted, 
                                          unsigned int *keysUnsorted, size_t len)
{
    int retval = 0;

    for(unsigned int i=0; i<len-1; ++i)
    {	   
        if( (keysSorted[i])>(keysSorted[i+1]) )
        {
            printf("Unordered key[%u]:%u > key[%u]:%u\n", i, keysSorted[i], i+1, keysSorted[i+1]);
            retval = 1;
            break;
        }		
    }

    if (valuesSorted)
    {
        for(unsigned int i=0; i<len; ++i)
        {
            if( keysUnsorted[(valuesSorted[i])] != keysSorted[i] )
            {
                printf("Incorrectly sorted value[%u] (%u) %u != %u\n", 
                       i, valuesSorted[i], keysUnsorted[valuesSorted[i]], keysSorted[i]);
                retval = 1;
                break;
            }
        }
    }

    return retval;
}

template<>
int SortSupport<float>::verifySort(float *keysSorted, unsigned int *valuesSorted, 
                                   float *keysUnsorted, size_t len)
{
    int retval = 0;

    for(unsigned int i=0; i<len-1; ++i)
    {	   
        if( (keysSorted[i])>(keysSorted[i+1]) )
        {
            printf("Unordered key[%u]:%f > key[%u]:%f\n", i, keysSorted[i], i+1, keysSorted[i+1]);
            retval = 1;
            break;
        }		
    }

    if (valuesSorted)
    {
        for(unsigned int i=0; i<len; ++i)
        {
            if( keysUnsorted[(valuesSorted[i])] != keysSorted[i] )
            {
                printf("Incorrectly sorted value[%u] (%u) %f != %f\n", 
                    i, valuesSorted[i], keysUnsorted[valuesSorted[i]], keysSorted[i]);
                retval = 1;
                break;
            }
        }
    }

    return retval;
}

template <typename T>
int radixSortTest(CUDPPHandle plan, CUDPPConfiguration config, size_t *tests, 
                  unsigned int numTests, size_t numElements, unsigned int keybits,
                  testrigOptions testOptions, bool quiet)
{
    int retval = 0;
    
    T *h_keys, *h_keysSorted, *d_keys;
    unsigned int *h_values, *h_valuesSorted, *d_values;

    char outString[100];
    sprintf(outString, "%s %s", config.datatype == CUDPP_FLOAT ? "float" : "unsigned int",
                                config.options == CUDPP_OPTION_KEYS_ONLY ? "keys" : "key-value pairs");

    h_keys       = (T*)malloc(numElements*sizeof(T));
    h_keysSorted = (T*)malloc(numElements*sizeof(T));
    h_values     = 0;			
    h_valuesSorted = 0;


    if (config.options & CUDPP_OPTION_KEY_VALUE_PAIRS)	    
    {
        h_values       = (unsigned int*)malloc(numElements*sizeof(unsigned int));
        h_valuesSorted = (unsigned int*)malloc(numElements*sizeof(unsigned int));

        for(unsigned int i=0; i < numElements; ++i)   			
            h_values[i] = i; 		
    }																	

    // Fill up with some random data   
    SortSupport<T>::fillVector(h_keys, numElements, keybits);		

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_keys, numElements*sizeof(T)));
    if (config.options & CUDPP_OPTION_KEY_VALUE_PAIRS)
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_values, numElements*sizeof(unsigned int)));
    else
        d_values = 0;

    // run multiple iterations to compute an average sort time
    cudaEvent_t start_event, stop_event;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );

    for (unsigned int k = 0; k < numTests; ++k)
    {
        if(numTests == 1)
            tests[0] = numElements;			  			

        if(!quiet)
            printf("Running a sort of %ld %s\n", tests[k], outString);        

        float totalTime = 0;

        for (int i = 0; i < testOptions.numIterations; i++)
        {
            CUDA_SAFE_CALL(cudaMemcpy(d_keys, h_keys, tests[k] * sizeof(T), cudaMemcpyHostToDevice));
            if(config.options & CUDPP_OPTION_KEY_VALUE_PAIRS)
            {
                CUDA_SAFE_CALL( cudaMemcpy((void*)d_values, 
                                           (void*)h_values, 
                                           tests[k] * sizeof(unsigned int), 
                                           cudaMemcpyHostToDevice) );
            }

            CUDA_SAFE_CALL( cudaEventRecord(start_event, 0) );

            cudppSort(plan, d_keys, (void*)d_values, keybits, tests[k]);            			 

            CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
            CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );

            float time = 0;
            CUDA_SAFE_CALL( cudaEventElapsedTime(&time, start_event, stop_event));
            totalTime += time;
        }
        
        CUT_CHECK_ERROR("testradixSort - cudppRadixSort");

        // copy results
        CUDA_SAFE_CALL(cudaMemcpy(h_keysSorted, d_keys, tests[k] * sizeof(T), cudaMemcpyDeviceToHost));
        if (config.options & CUDPP_OPTION_KEY_VALUE_PAIRS)
        {
            CUDA_SAFE_CALL( cudaMemcpy((void*)h_valuesSorted, 
                                       (void*)d_values, 
                                       tests[k] * sizeof(unsigned int), 
                                       cudaMemcpyDeviceToHost) );
        }
        else
            h_values = 0;	        

        retval += SortSupport<T>::verifySort(h_keysSorted, h_valuesSorted, h_keys, tests[k]);

        if(!quiet)
        {			  
            printf("%s test %s\n", testOptions.runMode, (retval == 0) ? "PASSED" : "FAILED");
            printf("Average execution time: %f ms\n", totalTime / testOptions.numIterations);
        }
        else
        {
            printf("\t%10ld\t%0.4f\n", tests[k], totalTime / testOptions.numIterations);
        }
    }
    printf("\n");


    CUT_CHECK_ERROR("after radixsort");

    cudaFree(d_keys);
    if (config.options & CUDPP_OPTION_KEY_VALUE_PAIRS)
        cudaFree(d_values);
    free(h_keys);
    free(h_values);	

    return retval;
}

/**
 * testRadixSort tests cudpp's radix sort
 * Possible command line arguments:
 * - --keysonly, tests only a set of keys
 * - --keyval, tests a set of keys with associated values
 * - --n=#, number of elements in sort
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @param configPtr Configuration for scan, set by caller
 * @return Number of tests that failed regression (0 for all pass)
 * @see cudppSort
*/
int testRadixSort(int argc, const char **argv, CUDPPConfiguration *configPtr)
{

    int cmdVal;
    int keybits = 32;
    int retval = 0;
    
    bool quiet;        
    char out[80];
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);        
    
    CUDPPConfiguration config;
    config.algorithm = CUDPP_SORT_RADIX;
    config.datatype = CUDPP_UINT;
    config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
	
    if(configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        config.datatype = CUDPP_UINT;       
    }
 
    size_t test[] = {39, 128, 256, 512, 513, 1000, 1024, 1025, 32768, 
                     45537, 65536, 131072, 262144, 500001, 524288, 
                     1048577, 1048576, 1048581, 2097152, 4194304, 
                     8388608};
    
    int numTests = sizeof(test)/sizeof(test[0]);
    
    
    size_t numElements = test[numTests - 1];

    if( cutCheckCmdLineFlag(argc, (const char**)argv, "help") )
    {
        printf("Command line:\nradixsort_block [-n=<number of elements>] [-keybits=<number of key bits>]\n");
        exit(1);
    }

    bool keysOnly = (cutCheckCmdLineFlag(argc, (const char**)argv, "keysonly") == CUTTrue);	
    bool keyValue = (cutCheckCmdLineFlag(argc, (const char**)argv, "keyval") == CUTTrue);
    quiet = (cutCheckCmdLineFlag(argc, (const char**)argv, "quiet") == CUTTrue);	
    
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "float") )
    {     
        config.datatype = CUDPP_FLOAT;
    }
    else if( cutCheckCmdLineFlag(argc, (const char**)argv, "uint") )
    {        
        config.datatype = CUDPP_UINT;
    }
   
    if(config.options == CUDPP_OPTION_KEYS_ONLY || keysOnly) 
    {
        keysOnly = true;        
        config.options = CUDPP_OPTION_KEYS_ONLY;
    }	
    else 
    {
        keysOnly = false;
        config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
    }

    if( cutGetCmdLineArgumenti( argc, (const char**)argv, "n", &cmdVal) )
    { 
        numElements = cmdVal;
        numTests = 1;                		
    }
    if( cutGetCmdLineArgumenti( argc, (const char**)argv, "keybits", &cmdVal) )
    {
        keybits = cmdVal;
    }    

    sprintf(out, "%s %s", config.datatype == CUDPP_FLOAT ? "float" : "unsigned int",
                          config.options == CUDPP_OPTION_KEYS_ONLY ? "keys" : "key-value pairs");
    
    CUDPPHandle plan;
    CUDPPResult result = CUDPP_SUCCESS;  

    result = cudppPlan(&plan, config, numElements, 1, 0);	


    if(result != CUDPP_SUCCESS)
    {
        printf("Error in plan creation\n");
        retval = numTests;
        cudppDestroyPlan(plan);
        return retval;
    }
        
    switch(config.datatype)
    {        
    case CUDPP_UINT:
        retval = radixSortTest<unsigned int>(plan, config, test, numTests, numElements, keybits, testOptions, quiet);
        break;
    case CUDPP_FLOAT:	
        retval = radixSortTest<float>(plan, config, test, numTests, numElements, keybits, testOptions, quiet);
        break;
    }

    result = cudppDestroyPlan(plan);
    
    if (result != CUDPP_SUCCESS)
    {	
        printf("Error destroying CUDPPPlan for Scan\n");
        retval = numTests;
    }
        	          
    return retval;
}
