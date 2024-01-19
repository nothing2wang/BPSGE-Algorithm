function out_value = solve_eq_3(cor_3,cor_2,cor_1,cor_0)

results = roots([cor_3 cor_2 cor_1 cor_0]);
re_idx = find(imag(results)==0);
re_vec = results(re_idx);
out_idx =find(re_vec>=0);
out_value = re_vec(out_idx);

end