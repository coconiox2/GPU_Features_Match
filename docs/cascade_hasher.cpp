(
				const std::vector<MatrixT> & vec_descriptions,
				const Eigen::VectorXf & zero_mean_descriptor,
				std::map<IndexT, HashedDescriptions>  & hashed_descriptions
)
{
					// Allocate space for hash codes.
					const typename MatrixT::Index nbDescriptions = descriptions.rows();
					hashed_descriptions.hashed_desc.resize(nbDescriptions);


					//整合所有的描述符放在一个矩阵里传到device
					/*descriptor = descriptions.row(i).template cast<float>();
					descriptor -= zero_mean_descriptor;*/
					Eigen::MatrixXf primary_projection;
					Eigen::MatrixXf descriptionsMat;
					descriptionsMat = descriptions.template cast<float>();
					for (int k = 0; k < descriptionsMat.rows(); k++) {
						descriptionsMat.row(k) -= zero_mean_descriptor;
					}


					const float *descriptionsMat_data_temp = descriptionsMat.data();
					const float *primary_hash_projection_data_temp = primary_hash_projection_.data();
					const float *secondary_hash_projection_data_temp = secondary_hash_projection_.data();

					float *descriptionsMat_data_temp_1 = (float*)malloc(descriptions.rows()*descriptions.cols() * sizeof(float));
					float *primary_hash_projection_data_temp_1 = (float*)malloc(primary_hash_projection_.rows()*primary_hash_projection_.cols() * sizeof(float));
					//float *secondary_hash_projection_data_temp_1 = (float )

					clock_t start = clock();
					//行优先转化为列优先
					RTOC sRTOC;
					sRTOC.cToR(descriptionsMat_data_temp, descriptionsMat.rows(), descriptionsMat.cols(), descriptionsMat_data_temp_1);
					sRTOC.cToR(primary_hash_projection_data_temp, primary_hash_projection_.rows(), primary_hash_projection_.cols(), primary_hash_projection_data_temp_1);
					//RTOC::cToR(descriptionsMat_data_temp, descriptionsMat.rows(), descriptionsMat.cols(), descriptionsMat_data_temp_1);
					//RTOC::cToR(primary_hash_projection_data_temp, primary_hash_projection_.rows(), primary_hash_projection_.cols(), primary_hash_projection_data_temp_1);

					const float *descriptionsMat_data = const_cast<const float*> (descriptionsMat_data_temp_1);
					const float *primary_hash_projection_data = const_cast<const float*>(primary_hash_projection_data_temp_1);
					//const Eigen::MatrixXf primary_projection = descriptionsMat * primary_hash_projection_;

					int descriptions_size = nbDescriptions * (descriptions.cols());
					int primary_hash_projection_size = (primary_hash_projection_.rows()) * (primary_hash_projection_.cols());
					int result_size = nbDescriptions * (primary_hash_projection_.cols());
					//CPU存放结果的指针
					float *mat_result_data = (float*)malloc(result_size * sizeof(float));
					/*
					** GPU 计算矩阵相乘
					*/
					dim3 threads(1, 1);
					dim3 grid(1, 1);
					// 创建并初始化 CUBLAS 库对象
					cublasHandle_t handle;
					int status = cublasCreate(&handle);

					if (status != CUBLAS_STATUS_SUCCESS)
					{
						if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
							std::cout << "CUBLAS 对象实例化出错" << std::endl;
						}
						getchar();

					}
					//GPU中存放计算矩阵和计算结果的指针
					float *descriptionsMat_data_device,
						*primary_hash_projection_data_device,
						*mat_result_device;
					cudaMalloc((void **)&descriptionsMat_data_device, sizeof(float) * descriptions_size);
					cudaMalloc((void **)&primary_hash_projection_data_device, sizeof(float) * primary_hash_projection_size);
					cudaMalloc((void **)&mat_result_device, sizeof(float) * result_size);

					// 将矩阵数据传递进 显存 中已经开辟好了的空间
					cudaMemcpy(descriptionsMat_data_device, descriptionsMat_data, sizeof(float) * descriptions_size, cudaMemcpyHostToDevice);
					cudaMemcpy(primary_hash_projection_data_device, primary_hash_projection_data, sizeof(float) * primary_hash_projection_size, cudaMemcpyHostToDevice);

					// 同步函数
					cudaThreadSynchronize();
					// 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
					float a = 1; float b = 0;
					// 矩阵相乘。该函数必然将数组解析成列优先数组
					status = cublasSgemm(
						handle,    // blas 库对象
						CUBLAS_OP_N,    // 矩阵 A 属性参数
						CUBLAS_OP_N,    // 矩阵 B 属性参数	
						primary_hash_projection_.cols(),    // B, C 的列数, n
						descriptions.rows(),    // A, C 的行数 m
						descriptions.cols(),    // A 的列数和 B 的行数 k
						&a,    // 运算式的 α 值
						primary_hash_projection_data_device,    // B 在显存中的地址
						primary_hash_projection_.cols(),    // B, C 的列数, n
						descriptionsMat_data_device,    // A 在显存中的地址
						descriptions.cols(),    // ldb k
						&b,    // 运算式的 β 值
						mat_result_device,    // C 在显存中的地址(结果矩阵)
						primary_hash_projection_.cols()    // B, C 的列数, n
					);
					cublasDestroy(handle);
					// 将在GPU端计算好的结果拷贝回CPU端
					//int result_size = nbDescriptions * (primary_hash_projection_.cols());
					status = cublasGetMatrix(
						nbDescriptions,//m
						primary_hash_projection_.cols(),//n
						sizeof(*mat_result_data),
						mat_result_device,//d_c
						nbDescriptions,//m
						mat_result_data,//c
						nbDescriptions//m
					);


					for (int i = 0; i < descriptions.rows(); ++i) {
						// Allocate space for each bucket id.
						hashed_descriptions.hashed_desc[i].bucket_ids.resize(nb_bucket_groups_);
						// Compute hash code.
						auto& hash_code = hashed_descriptions.hashed_desc[i].hash_code;
						hash_code = stl::dynamic_bitset(descriptions.cols());
						for (int j = 0; j < nb_hash_code_; ++j)
						{
							hash_code[j] = mat_result_data[(i*nb_hash_code_ + j)] > 0;
						}
					}
					//计时
					clock_t end = clock();
					double endtime = (double)(end - start) / CLOCKS_PER_SEC;

					std::cout << "Total time:" << endtime << std::endl;		//s为单位
					std::cout << "Total time:" << endtime * 1000 << "ms" << std::endl;	//ms为单位

																						// 清理掉使用过的内存

					free(mat_result_data);
					cudaFree(descriptionsMat_data_device);
					cudaFree(primary_hash_projection_data_device);
					cudaFree(mat_result_device);

					// 释放 CUBLAS 库对象
					cublasDestroy(handle);

					// Determine the bucket index for each group.
					Eigen::MatrixXf secondary_projection;
					for (int j = 0; j < nb_bucket_groups_; ++j)
					{
						uint16_t bucket_id = 0;
						secondary_projection = secondary_hash_projection_[j] * descriptionsMat;

						//有待测试
						for (int h = 0; h < descriptionsMat.rows(); h++) {
							for (int k = 0; k < nb_bits_per_bucket_; ++k)
							{
								bucket_id = (bucket_id << 1) + (secondary_projection(h, k) > 0 ? 1 : 0);
							}

							hashed_descriptions.hashed_desc[h].bucket_ids[j] = bucket_id;
						}
					}


				}