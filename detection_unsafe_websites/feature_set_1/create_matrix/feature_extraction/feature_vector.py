from feature_extraction.feature_class import UrlJsonClass
import json


def get_feature_vector(path=None, json_data=None) -> tuple:

    if json_data is None:
        with open(path) as f:
            json_data = json.load(f)
        f.close()

    """
    Empty json -> return false.
    """
    try:
        message = json_data['message']
        print('Empty json: {}'.format(message))
        return False, [], 4
    except KeyError:
        pass

    # try:
    #   if 'ERR_NAME_NOT_RESOLVED' in str(json_data['data']):
    #       return False, [], 5
    #   else:
    #       pass
    # except KeyError:
    #     return False, [], 6

    try:
        _ = json_data['data']
    except KeyError:
        return False, [], 6

    feature_list = []
    features = UrlJsonClass(json_data)

    # feature_list.append(features.get_1_number_ips())
    # feature_list.append(features.get_2_number_countries())
    # feature_list.append(features.get_3_number_asns())
    # feature_list.append(features.get_4_mean_asns())
    # feature_list.append(features.get_5_std_asns())
    # feature_list.append(features.get_6_number_domains())
    # feature_list.append(features.get_7_diff_domains())
    # feature_list.append(features.get_8_servers())
    # feature_list.append(features.get_9_urls())
    # feature_list.append(features.get_10_diff_urls())
    # feature_list.append(features.get_11_https_in_url())
    # feature_list.append(features.get_12_domain_in_url())
    # feature_list.append(features.get_13_image_in_url())
    # feature_list.append(features.get_14_sizemean_url())
    # feature_list.append(features.get_15_sizestd_url())
    # feature_list.append(features.get_16_javascript_url())
    # feature_list.append(features.get_17_javascript_url())
    # feature_list.append(features.get_18_length_linkdomains())
    # feature_list.append(features.get_19_difftld_linkdomains())
    # feature_list.append(features.get_20_numsubdomainmean_linkdomains())
    # feature_list.append(features.get_21_numsubdomainstd_linkdomains())
    # feature_list.append(features.get_22_numcertificates_linkdomains())
    # feature_list.append(features.get_23_certicatevalidationmean_linkdomains())
    # feature_list.append(features.get_24_certicatevalidationstd_linkdomains())
    # feature_list.append(features.get_25_certicatevalidation2mean_linkdomains())
    # feature_list.append(features.get_26_certicatevalidation2std_linkdomains())
    # feature_list.append(features.get_27_diff_countries())
    # feature_list.append(features.get_28_confidence_mean())
    # feature_list.append(features.get_29_confidence_std())
    # feature_list.append(features.get_30_priority_mean())
    # feature_list.append(features.get_31_priority_std())
    # feature_list.append(features.get_32_abp_len())
    # feature_list.append(features.get_33_js_len())
    # feature_list.append(features.get_34_cookie_len())
    # feature_list.append(features.get_35_url_mean())
    # feature_list.append(features.get_36_url_std())
    # feature_list.append(features.get_37_count_mean())
    # feature_list.append(features.get_38_count_std())
    # feature_list.append(features.get_39_size_mean())
    # feature_list.append(features.get_40_size_std())
    # feature_list.append(features.get_41_ensize_mean())
    # feature_list.append(features.get_42_ensize_std())
    # feature_list.append(features.get_43_latency_mean())
    # feature_list.append(features.get_44_latency_std())
    # feature_list.append(features.get_45_compression_mean())
    # feature_list.append(features.get_46_compression_std())
    # feature_list.append(features.get_47_ips_mean())
    # feature_list.append(features.get_48_ips_std())
    # feature_list.append(features.get_49_countries_mean())
    # feature_list.append(features.get_50_countries_std())
    # feature_list.append(features.get_51_count_mean())
    # feature_list.append(features.get_52_count_std())
    # feature_list.append(features.get_53_size_mean())
    # feature_list.append(features.get_54_size_std())
    # feature_list.append(features.get_55_ensize_mean())
    # feature_list.append(features.get_56_ensize_std())
    # feature_list.append(features.get_57_ips_mean())
    # feature_list.append(features.get_58_ips_std())
    # feature_list.append(features.get_59_countries_mean())
    # feature_list.append(features.get_60_countries_std())
    # feature_list.append(features.get_61_count_mean())
    # feature_list.append(features.get_62_count_std())
    # feature_list.append(features.get_63_size_mean())
    # feature_list.append(features.get_64_size_std())
    # feature_list.append(features.get_65_ensize_mean())
    # feature_list.append(features.get_66_ensize_std())
    # feature_list.append(features.get_67_ips_mean())
    # feature_list.append(features.get_68_ips_std())
    # feature_list.append(features.get_69_countries_mean())
    # feature_list.append(features.get_70_countries_std())
    # feature_list.append(features.get_71_protocols_mean())
    # feature_list.append(features.get_72_protocols_std())
    # feature_list.append(features.get_73_count_mean())
    # feature_list.append(features.get_74_count_std())
    # feature_list.append(features.get_75_size_mean())
    # feature_list.append(features.get_76_size_std())
    # feature_list.append(features.get_77_ensize_mean())
    # feature_list.append(features.get_78_ensize_std())
    # feature_list.append(features.get_79_ips_mean())
    # feature_list.append(features.get_80_ips_std())
    # feature_list.append(features.get_81_countries_mean())
    # feature_list.append(features.get_82_countries_std())
    # feature_list.append(features.get_83_count_mean())
    # feature_list.append(features.get_84_count_std())
    # feature_list.append(features.get_85_size_mean())
    # feature_list.append(features.get_86_size_std())
    # feature_list.append(features.get_87_ensize_mean())
    # feature_list.append(features.get_88_ensize_std())
    # feature_list.append(features.get_89_ips_mean())
    # feature_list.append(features.get_90_ips_std())
    # feature_list.append(features.get_91_countries_mean())
    # feature_list.append(features.get_92_countries_std())
    # feature_list.append(features.get_93_redirects_mean())
    # feature_list.append(features.get_94_redirects_std())
    # feature_list.append(features.get_95_initiators_mean())
    # feature_list.append(features.get_96_initiators_std())
    # feature_list.append(features.get_97_domain_in_initiators_mean())
    # feature_list.append(features.get_98_domain_in_initiators_std())
    # feature_list.append(features.get_99_count_mean())
    # feature_list.append(features.get_100_count_std())
    # feature_list.append(features.get_101_size_mean())
    # feature_list.append(features.get_102_size_std())
    # feature_list.append(features.get_103_ensize_mean())
    # feature_list.append(features.get_104_ensize_std())
    # feature_list.append(features.get_105_ips_mean())
    # feature_list.append(features.get_106_ips_std())
    # feature_list.append(features.get_107_countries_mean())
    # feature_list.append(features.get_108_countries_std())
    # feature_list.append(features.get_109_redirects_mean())
    # feature_list.append(features.get_110_redirects_std())
    # feature_list.append(features.get_111_subDomains_mean())
    # feature_list.append(features.get_112_subDomains_std())
    # feature_list.append(features.get_113_size_mean())
    # feature_list.append(features.get_114_size_std())
    # feature_list.append(features.get_115_ensize_mean())
    # feature_list.append(features.get_116_ensize_std())
    # feature_list.append(features.get_117_countries_mean())
    # feature_list.append(features.get_118_countries_std())
    # feature_list.append(features.get_119_redirects_mean())
    # feature_list.append(features.get_120_redirects_std())
    # feature_list.append(features.get_121_ipv6_mean())
    # feature_list.append(features.get_122_ipv6_std())
    # feature_list.append(features.get_123_ipv6_mean())
    # feature_list.append(features.get_124_ipv6_std())
    # feature_list.append(features.get_125_domains_mean())
    # feature_list.append(features.get_126_domains_std())
    # feature_list.append(features.get_127_secureRequests())
    # feature_list.append(features.get_128_securePercentage())
    # feature_list.append(features.get_129_IPv6Percentage())
    # feature_list.append(features.get_130_uniqCountries())
    # feature_list.append(features.get_131_totalLinks())
    # feature_list.append(features.get_132_adBlocked())
    # # ['data']
    # feature_list.append(features.get_133_len_request())
    # feature_list.append(features.get_134_domain_in_docuurl_list_mean())
    # feature_list.append(features.get_135_domain_in_docuurl_list_std())
    # feature_list.append(features.get_136_upgrade_insecure_requests_mean())
    # feature_list.append(features.get_137_upgrade_insecure_requests_std())
    # feature_list.append(features.get_138_status_list_mean())
    # feature_list.append(features.get_139_status_list_std())
    # feature_list.append(features.get_140_content_length_mean())
    # feature_list.append(features.get_141_content_length_std())
    # feature_list.append(features.get_142_encodedDataLength_mean())
    # feature_list.append(features.get_143_encodedDataLength_std())
    # # feature_list.append(features.get_144_requestTime_mean())
    # # feature_list.append(features.get_145_requestTime_std())
    # # feature_list.append(features.get_146_proxyStart_mean())
    # # feature_list.append(features.get_147_proxyStart_std())
    # # feature_list.append(features.get_148_proxyEnd_mean())
    # # feature_list.append(features.get_149_proxyEnd_std())
    # # feature_list.append(features.get_150_dnsStart_mean())
    # # feature_list.append(features.get_151_dnsStart_std())
    # # feature_list.append(features.get_152_dnsEnd_mean())
    # # feature_list.append(features.get_153_dnsEnd_std())
    # # feature_list.append(features.get_154_connectStart_mean())
    # # feature_list.append(features.get_155_connectStart_std())
    # # feature_list.append(features.get_156_connectEnd_mean())
    # # feature_list.append(features.get_157_connectEnd_std())
    # # feature_list.append(features.get_158_sslStart_mean())
    # # feature_list.append(features.get_159_sslStart_std())
    # # feature_list.append(features.get_160_sslEnd_mean())
    # # feature_list.append(features.get_161_sslEnd_std())
    # # feature_list.append(features.get_162_workerStart_mean())
    # # feature_list.append(features.get_163_workerStart_std())
    # # feature_list.append(features.get_164_workerReady_mean())
    # # feature_list.append(features.get_165_workerReady_std())
    # # feature_list.append(features.get_166_sendStart_mean())
    # # feature_list.append(features.get_167_sendStart_std())
    # # feature_list.append(features.get_168_sendEnd_mean())
    # # feature_list.append(features.get_169_sendEnd_std())
    # # feature_list.append(features.get_170_pushStart_mean())
    # # feature_list.append(features.get_171_pushStart_std())
    # # feature_list.append(features.get_172_pushEnd_mean())
    # # feature_list.append(features.get_173_pushEnd_std())
    # # feature_list.append(features.get_174_receiveHeadersEnd_mean())
    # # feature_list.append(features.get_175_receiveHeadersEnd_std())
    # feature_list.append(features.get_176_len_request_list_mean())
    # feature_list.append(features.get_177_len_request_list_std())
    # feature_list.append(features.get_178_response_encodedDataLength_mean())
    # feature_list.append(features.get_179_response_encodedDataLength_std())
    # feature_list.append(features.get_180_response_dataLength_mean())
    # feature_list.append(features.get_181_response_dataLength_std())
    # feature_list.append(features.get_182_response_respo_encodedDataLength_mean())
    # feature_list.append(features.get_183_response_respo_encodedDataLength_std())
    # # feature_list.append(features.get_184_response_requestTime_mean())
    # # feature_list.append(features.get_185_response_requestTime_std())
    # # feature_list.append(features.get_186_response_proxyStart_mean())
    # # feature_list.append(features.get_187_response_proxyStart_std())
    # # feature_list.append(features.get_188_response_proxyEnd_mean())
    # # feature_list.append(features.get_189_response_proxyEnd_std())
    # # feature_list.append(features.get_190_response_dnsStart_mean())
    # # feature_list.append(features.get_191_response_dnsStart_std())
    # # feature_list.append(features.get_192_response_dnsEnd_mean())
    # # feature_list.append(features.get_193_response_dnsEnd_std())
    # # feature_list.append(features.get_194_response_connectStart_mean())
    # # feature_list.append(features.get_195_response_connectStart_std())
    # # feature_list.append(features.get_196_response_connectEnd_mean())
    # # feature_list.append(features.get_197_response_connectEnd_std())
    # # feature_list.append(features.get_198_response_sslStart_mean())
    # # feature_list.append(features.get_199_response_sslStart_std())
    # # feature_list.append(features.get_200_response_sslEnd_mean())
    # # feature_list.append(features.get_201_response_sslEnd_std())
    # # feature_list.append(features.get_202_response_workerStart_mean())
    # # feature_list.append(features.get_203_response_workerStart_std())
    # # feature_list.append(features.get_204_response_workerReady_mean())
    # # feature_list.append(features.get_205_response_workerReady_std())
    # # feature_list.append(features.get_206_response_sendStart_mean())
    # # feature_list.append(features.get_207_response_sendStart_std())
    # # feature_list.append(features.get_208_response_sendEnd_mean())
    # # feature_list.append(features.get_209_response_sendEnd_std())
    # feature_list.append(features.get_210_response_pushStart_mean())
    # feature_list.append(features.get_211_response_pushStart_std())
    # feature_list.append(features.get_212_response_pushEnd_mean())
    # feature_list.append(features.get_213_response_pushEnd_std())
    # feature_list.append(features.get_214_response_receiveHeadersEnd_mean())
    # feature_list.append(features.get_215_response_receiveHeadersEnd_std())
    # feature_list.append(features.get_216_securityState_mean())
    # feature_list.append(features.get_217_securityState_std())
    # feature_list.append(features.get_218_sanList_mean())
    # feature_list.append(features.get_219_sanList_std())
    # feature_list.append(features.get_220_subject_name_in_san_list_mean())
    # feature_list.append(features.get_221_subject_name_in_san_list_std())
    # feature_list.append(features.get_222_cert_valid_mean())
    # feature_list.append(features.get_223_cert_valid_std())
    # feature_list.append(features.get_224_cert_valid_now_mean())
    # feature_list.append(features.get_225_cert_valid_now_std())
    # feature_list.append(features.get_226_hashes_list_mean())
    # feature_list.append(features.get_227_hashes_list_std())
    # feature_list.append(features.get_228_hashes_list_mean())
    # feature_list.append(features.get_229_cookie_expires_mean())
    # feature_list.append(features.get_230_cookie_expires_std())
    # feature_list.append(features.get_231_cookie_expires_now_mean())
    # feature_list.append(features.get_232_cookie_expires_now_std())
    # feature_list.append(features.get_233_cookies_sizes_mean())
    # feature_list.append(features.get_234_cookies_sizes_std())
    # feature_list.append(features.get_235_cookie_http_only_mean())
    # feature_list.append(features.get_236_cookie_http_only_std())
    # feature_list.append(features.get_237_cookie_secure_mean())
    # feature_list.append(features.get_238_cookie_secure_std())
    # feature_list.append(features.get_239_cookie_session_mean())
    # feature_list.append(features.get_240_cookie_session_std())
    # feature_list.append(features.get_241_len_links())
    # feature_list.append(features.get_242_links_domain_in_url_mean())
    # feature_list.append(features.get_243_links_domain_in_url_std())
    # feature_list.append(features.get_244_diff_tld())
    # feature_list.append(features.get_245_diff_tld())
    # feature_list.append(features.get_246_diff_globals())











    feature_list.append(features.get_1_number_ips())
    feature_list.append(features.get_2_number_countries())
    feature_list.append(features.get_3_number_asns())
    feature_list.append(features.get_4_mean_asns())
    feature_list.append(features.get_5_std_asns())
    feature_list.append(features.get_6_number_domains())
    feature_list.append(features.get_7_diff_domains())
    feature_list.append(features.get_8_servers())
    feature_list.append(features.get_9_urls())
    feature_list.append(features.get_10_diff_urls())
    feature_list.append(features.get_11_https_in_url())
    feature_list.append(features.get_12_domain_in_url())
    feature_list.append(features.get_13_image_in_url())
    feature_list.append(features.get_14_sizemean_url())
    feature_list.append(features.get_15_sizestd_url())
    feature_list.append(features.get_16_javascript_url())
    feature_list.append(features.get_17_javascript_url())
    feature_list.append(features.get_18_length_linkdomains())
    feature_list.append(features.get_19_difftld_linkdomains())
    feature_list.append(features.get_20_numsubdomainmean_linkdomains())
    feature_list.append(features.get_21_numsubdomainstd_linkdomains())
    feature_list.append(features.get_22_numcertificates_linkdomains())
    feature_list.append(features.get_23_certicatevalidationmean_linkdomains())
    feature_list.append(features.get_24_certicatevalidationstd_linkdomains())
    feature_list.append(features.get_25_certicatevalidation2mean_linkdomains())
    feature_list.append(features.get_26_certicatevalidation2std_linkdomains())
    feature_list.append(features.get_27_diff_countries())
    feature_list.append(features.get_28_confidence_mean())
    feature_list.append(features.get_29_confidence_std())
    feature_list.append(features.get_30_priority_mean())
    feature_list.append(features.get_31_priority_std())
    feature_list.append(features.get_32_abp_len())
    feature_list.append(features.get_33_js_len())
    feature_list.append(features.get_34_cookie_len())
    feature_list.append(features.get_35_url_mean())
    feature_list.append(features.get_36_url_std())
    feature_list.append(features.get_37_count_mean())
    feature_list.append(features.get_38_count_std())
    feature_list.append(features.get_39_size_mean())
    feature_list.append(features.get_40_size_std())
    feature_list.append(features.get_41_ensize_mean())
    feature_list.append(features.get_42_ensize_std())
    feature_list.append(features.get_43_latency_mean())
    feature_list.append(features.get_44_latency_std())
    feature_list.append(features.get_45_compression_mean())
    feature_list.append(features.get_46_compression_std())
    feature_list.append(features.get_47_ips_mean())
    feature_list.append(features.get_48_ips_std())
    feature_list.append(features.get_49_countries_mean())
    feature_list.append(features.get_50_countries_std())
    feature_list.append(features.get_51_count_mean())
    feature_list.append(features.get_52_count_std())
    feature_list.append(features.get_53_size_mean())
    feature_list.append(features.get_54_size_std())
    feature_list.append(features.get_55_ensize_mean())
    feature_list.append(features.get_56_ensize_std())
    feature_list.append(features.get_57_ips_mean())
    feature_list.append(features.get_58_ips_std())
    feature_list.append(features.get_59_countries_mean())
    feature_list.append(features.get_60_countries_std())
    feature_list.append(features.get_61_count_mean())
    feature_list.append(features.get_62_count_std())
    feature_list.append(features.get_63_size_mean())
    feature_list.append(features.get_64_size_std())
    feature_list.append(features.get_65_ensize_mean())
    feature_list.append(features.get_66_ensize_std())
    feature_list.append(features.get_67_ips_mean())
    feature_list.append(features.get_68_ips_std())
    feature_list.append(features.get_69_countries_mean())
    feature_list.append(features.get_70_countries_std())
    feature_list.append(features.get_71_protocols_mean())
    feature_list.append(features.get_72_protocols_std())
    feature_list.append(features.get_73_count_mean())
    feature_list.append(features.get_74_count_std())
    feature_list.append(features.get_75_size_mean())
    feature_list.append(features.get_76_size_std())
    feature_list.append(features.get_77_ensize_mean())
    feature_list.append(features.get_78_ensize_std())
    feature_list.append(features.get_79_ips_mean())
    feature_list.append(features.get_80_ips_std())
    feature_list.append(features.get_81_countries_mean())
    feature_list.append(features.get_82_countries_std())
    feature_list.append(features.get_83_count_mean())
    feature_list.append(features.get_84_count_std())
    feature_list.append(features.get_85_size_mean())
    feature_list.append(features.get_86_size_std())
    feature_list.append(features.get_87_ensize_mean())
    feature_list.append(features.get_88_ensize_std())
    feature_list.append(features.get_89_ips_mean())
    feature_list.append(features.get_90_ips_std())
    feature_list.append(features.get_91_countries_mean())
    feature_list.append(features.get_92_countries_std())
    feature_list.append(features.get_93_redirects_mean())
    feature_list.append(features.get_94_redirects_std())
    feature_list.append(features.get_95_initiators_mean())
    feature_list.append(features.get_96_initiators_std())
    feature_list.append(features.get_97_domain_in_initiators_mean())
    feature_list.append(features.get_98_domain_in_initiators_std())
    feature_list.append(features.get_99_count_mean())
    feature_list.append(features.get_100_count_std())
    feature_list.append(features.get_101_size_mean())
    feature_list.append(features.get_102_size_std())
    feature_list.append(features.get_103_ensize_mean())
    feature_list.append(features.get_104_ensize_std())
    feature_list.append(features.get_105_ips_mean())
    feature_list.append(features.get_106_ips_std())
    feature_list.append(features.get_107_countries_mean())
    feature_list.append(features.get_108_countries_std())
    feature_list.append(features.get_109_redirects_mean())
    feature_list.append(features.get_110_redirects_std())
    feature_list.append(features.get_111_subDomains_mean())
    feature_list.append(features.get_112_subDomains_std())
    feature_list.append(features.get_113_size_mean())
    feature_list.append(features.get_114_size_std())
    feature_list.append(features.get_115_ensize_mean())
    feature_list.append(features.get_116_ensize_std())
    feature_list.append(features.get_117_countries_mean())
    feature_list.append(features.get_118_countries_std())
    feature_list.append(features.get_119_redirects_mean())
    feature_list.append(features.get_120_redirects_std())
    feature_list.append(features.get_121_ipv6_mean())
    feature_list.append(features.get_122_ipv6_std())
    feature_list.append(features.get_123_ipv6_mean())
    feature_list.append(features.get_124_ipv6_std())
    feature_list.append(features.get_125_domains_mean())
    feature_list.append(features.get_126_domains_std())
    feature_list.append(features.get_127_secureRequests())
    feature_list.append(features.get_128_securePercentage())
    feature_list.append(features.get_129_IPv6Percentage())
    feature_list.append(features.get_130_uniqCountries())
    feature_list.append(features.get_131_totalLinks())
    feature_list.append(features.get_132_adBlocked())
    # ['data']
    feature_list.append(features.get_133_len_request())
    feature_list.append(features.get_134_domain_in_docuurl_list_mean())
    feature_list.append(features.get_135_domain_in_docuurl_list_std())
    feature_list.append(features.get_136_upgrade_insecure_requests_mean())
    feature_list.append(features.get_137_upgrade_insecure_requests_std())
    feature_list.append(features.get_138_status_list_mean())
    feature_list.append(features.get_139_status_list_std())
    feature_list.append(features.get_140_content_length_mean())
    feature_list.append(features.get_141_content_length_std())
    feature_list.append(features.get_142_encodedDataLength_mean())
    feature_list.append(features.get_143_encodedDataLength_std())
    feature_list.append(features.get_144_requestTime_mean())
    feature_list.append(features.get_145_requestTime_std())
    feature_list.append(features.get_146_proxyStart_mean())
    feature_list.append(features.get_147_proxyStart_std())
    feature_list.append(features.get_148_proxyEnd_mean())
    feature_list.append(features.get_149_proxyEnd_std())
    feature_list.append(features.get_150_dnsStart_mean())
    feature_list.append(features.get_151_dnsStart_std())
    feature_list.append(features.get_152_dnsEnd_mean())
    feature_list.append(features.get_153_dnsEnd_std())
    feature_list.append(features.get_154_connectStart_mean())
    feature_list.append(features.get_155_connectStart_std())
    feature_list.append(features.get_156_connectEnd_mean())
    feature_list.append(features.get_157_connectEnd_std())
    feature_list.append(features.get_158_sslStart_mean())
    feature_list.append(features.get_159_sslStart_std())
    feature_list.append(features.get_160_sslEnd_mean())
    feature_list.append(features.get_161_sslEnd_std())
    feature_list.append(features.get_162_workerStart_mean())
    feature_list.append(features.get_163_workerStart_std())
    feature_list.append(features.get_164_workerReady_mean())
    feature_list.append(features.get_165_workerReady_std())
    feature_list.append(features.get_166_sendStart_mean())
    feature_list.append(features.get_167_sendStart_std())
    feature_list.append(features.get_168_sendEnd_mean())
    feature_list.append(features.get_169_sendEnd_std())
    feature_list.append(features.get_170_pushStart_mean())
    feature_list.append(features.get_171_pushStart_std())
    feature_list.append(features.get_172_pushEnd_mean())
    feature_list.append(features.get_173_pushEnd_std())
    feature_list.append(features.get_174_receiveHeadersEnd_mean())
    feature_list.append(features.get_175_receiveHeadersEnd_std())
    feature_list.append(features.get_176_len_request_list_mean())
    feature_list.append(features.get_177_len_request_list_std())
    feature_list.append(features.get_178_response_encodedDataLength_mean())
    feature_list.append(features.get_179_response_encodedDataLength_std())
    feature_list.append(features.get_180_response_dataLength_mean())
    feature_list.append(features.get_181_response_dataLength_std())
    feature_list.append(features.get_182_response_respo_encodedDataLength_mean())
    feature_list.append(features.get_183_response_respo_encodedDataLength_std())
    feature_list.append(features.get_184_response_requestTime_mean())
    feature_list.append(features.get_185_response_requestTime_std())
    feature_list.append(features.get_186_response_proxyStart_mean())
    feature_list.append(features.get_187_response_proxyStart_std())
    feature_list.append(features.get_188_response_proxyEnd_mean())
    feature_list.append(features.get_189_response_proxyEnd_std())
    feature_list.append(features.get_190_response_dnsStart_mean())
    feature_list.append(features.get_191_response_dnsStart_std())
    feature_list.append(features.get_192_response_dnsEnd_mean())
    feature_list.append(features.get_193_response_dnsEnd_std())
    feature_list.append(features.get_194_response_connectStart_mean())
    feature_list.append(features.get_195_response_connectStart_std())
    feature_list.append(features.get_196_response_connectEnd_mean())
    feature_list.append(features.get_197_response_connectEnd_std())
    feature_list.append(features.get_198_response_sslStart_mean())
    feature_list.append(features.get_199_response_sslStart_std())
    feature_list.append(features.get_200_response_sslEnd_mean())
    feature_list.append(features.get_201_response_sslEnd_std())
    feature_list.append(features.get_202_response_workerStart_mean())
    feature_list.append(features.get_203_response_workerStart_std())
    feature_list.append(features.get_204_response_workerReady_mean())
    feature_list.append(features.get_205_response_workerReady_std())
    feature_list.append(features.get_206_response_sendStart_mean())
    feature_list.append(features.get_207_response_sendStart_std())
    feature_list.append(features.get_208_response_sendEnd_mean())
    feature_list.append(features.get_209_response_sendEnd_std())
    feature_list.append(features.get_210_response_pushStart_mean())
    feature_list.append(features.get_211_response_pushStart_std())
    feature_list.append(features.get_212_response_pushEnd_mean())
    feature_list.append(features.get_213_response_pushEnd_std())
    feature_list.append(features.get_214_response_receiveHeadersEnd_mean())
    feature_list.append(features.get_215_response_receiveHeadersEnd_std())
    feature_list.append(features.get_216_securityState_mean())
    feature_list.append(features.get_217_securityState_std())
    feature_list.append(features.get_218_sanList_mean())
    feature_list.append(features.get_219_sanList_std())
    feature_list.append(features.get_220_subject_name_in_san_list_mean())
    feature_list.append(features.get_221_subject_name_in_san_list_std())
    feature_list.append(features.get_222_cert_valid_mean())
    feature_list.append(features.get_223_cert_valid_std())
    feature_list.append(features.get_224_cert_valid_now_mean())
    feature_list.append(features.get_225_cert_valid_now_std())
    feature_list.append(features.get_226_hashes_list_mean())
    feature_list.append(features.get_227_hashes_list_std())
    feature_list.append(features.get_228_hashes_list_mean())
    feature_list.append(features.get_229_cookie_expires_mean())
    feature_list.append(features.get_230_cookie_expires_std())
    feature_list.append(features.get_231_cookie_expires_now_mean())
    feature_list.append(features.get_232_cookie_expires_now_std())
    feature_list.append(features.get_233_cookies_sizes_mean())
    feature_list.append(features.get_234_cookies_sizes_std())
    feature_list.append(features.get_235_cookie_http_only_mean())
    feature_list.append(features.get_236_cookie_http_only_std())
    feature_list.append(features.get_237_cookie_secure_mean())
    feature_list.append(features.get_238_cookie_secure_std())
    feature_list.append(features.get_239_cookie_session_mean())
    feature_list.append(features.get_240_cookie_session_std())
    feature_list.append(features.get_241_len_links())
    feature_list.append(features.get_242_links_domain_in_url_mean())
    feature_list.append(features.get_243_links_domain_in_url_std())
    feature_list.append(features.get_244_diff_tld())
    feature_list.append(features.get_245_diff_tld())
    feature_list.append(features.get_246_diff_globals())


    # for i, feature in enumerate(feature_list):
    #     print('{}   {}'.format(i+1, feature))
    #
    # print('Number of features: {}'.format(len(feature_list)))

    return True, feature_list, 0


# if __name__ == '__main__':
#     get_feature_vector()