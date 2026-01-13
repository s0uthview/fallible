use syn::{parse_macro_input, ItemFn, ReturnType, Type, GenericArgument, PathArguments};
use proc_macro::TokenStream;
use quote::quote;

fn extract_result_error_type(return_type: &ReturnType) -> Option<&Type> {
    if let ReturnType::Type(_, ty) = return_type {
        if let Type::Path(type_path) = &**ty {
            if let Some(segment) = type_path.path.segments.last() {
                if segment.ident == "Result" {
                    if let PathArguments::AngleBracketed(args) = &segment.arguments {
                        if args.args.len() == 2 {
                            if let Some(GenericArgument::Type(err_type)) = args.args.iter().nth(1) {
                                return Some(err_type);
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

#[proc_macro_attribute]
pub fn fallible(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    let sig = &input.sig;
    let block = &input.block;
    let vis = &input.vis;

    let fn_name = sig.ident.to_string();
    let id_hash = fxhash::hash32(fn_name.as_bytes());

    let error_type = extract_result_error_type(&sig.output);

    let expanded = if let Some(err_ty) = error_type {
        quote! {
            #vis #sig {
                #[cfg(feature = "fallible-sim")]
                if ::fallible::fallible_core::should_simulate_failure(
                    ::fallible::fallible_core::FailurePoint {
                        id: ::fallible::fallible_core::FailurePointId(#id_hash),
                        function: #fn_name,
                        file: file!(),
                        line: line!(),
                        column: column!(),
                    }
                ) {
                    return Err(<#err_ty as ::fallible::fallible_core::FallibleError>::simulated_failure());
                }

                #block
            }
        }
    } else {
        quote! {
            #vis #sig #block
        }
    };

    expanded.into()
}